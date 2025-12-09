from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from typing import Dict, Optional, Set, Tuple

import networkx as nx
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.codegen_agent import ThreatCodegenAgent
from agents.congestion_agent import CongestionCollapseAgent
from agents.evaluators import SimpleImpactEvaluator
from agents.protocol_agent import ProtocolAttackAgent
from agents.satellite_agent import SatelliteDamageAgent
from llm_client import LLMScenarioGenerator
from simulation.solar_storm_model import SolarStormNodeOutageModel
from simulation.environment import LEONetworkModel
from simulation.evaluation import (
    EvaluationRecorder,
    EvaluationRow,
    compute_agent_metrics,
    compute_performance_metrics,
    summarize_delta,
)
from simulation.leocraft_starlink import flatten_performance_snapshot
from simulation.robustness_metrics import RobustnessEvaluator
from simulation.multi_agent_manager import MultiAgentManager
from simulation.scenario_repository import ScenarioRepository
from threat_scenarios.base import ScenarioContext


# Registry of supported threat models so they can be constructed from JSON configs.
THREAT_REGISTRY = {
    "solar_storm_node_outage": SolarStormNodeOutageModel,
    # "congestion_attack": CongestionModel,
    # "protocol_attack": ProtocolAttackModel,
}


def build_context() -> ScenarioContext:
    """Construct a default scenario context for demonstration purposes."""
    return ScenarioContext(
        satellite_count=120,
        inter_satellite_links=180,
        ground_stations=25,
        critical_services=["navigation", "earth-observation", "broadband"],
    )


def load_threat_json(config_path: Path) -> list[dict]:
    """Load threat model configuration from a JSON document.

    The loader is tolerant of two shapes:

    1) A list of threat configuration dictionaries, each containing ``type`` and
       optional ``params`` keys (preferred).
    2) A legacy wrapper object where the list is nested under a ``threats`` key
       or the document itself represents a single threat configuration.
    """

    if not config_path.exists():
        return []

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "threats" in payload and isinstance(payload["threats"], list):
            return payload["threats"]
        if "type" in payload:
            return [payload]
    raise ValueError(f"Threat JSON格式不符合要求: {config_path}")


def build_threat_models(threat_configs: list[dict]):
    """
    根据 multi-agent 生成的 JSON 配置，构建本次仿真的威胁模型实例列表。
    """

    models = []
    for cfg in threat_configs:
        ttype = cfg["type"]
        params = cfg.get("params", {})
        if ttype not in THREAT_REGISTRY:
            raise ValueError(f"Unknown threat type: {ttype}")
        cls = THREAT_REGISTRY[ttype]
        model = cls(**params)
        models.append(model)
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory where generated scenarios and network logs will be stored.",
    )
    parser.add_argument(
        "--network-name",
        default="LEO-Mesh-Alpha",
        help="Name of the simulated LEO network instance.",
    )
    return parser.parse_args()

def log_status(message: str) -> None:
    print(f"[STATUS] {message}")

def _extract_total_throughput(flat_snapshot: Optional[Dict[str, float]]) -> float:
    throughput_values = [
        value for key, value in (flat_snapshot or {}).items() if "throughput" in key.lower()
    ]
    if throughput_values:
        return float(sum(throughput_values))
    return float(sum((flat_snapshot or {}).values())) if flat_snapshot else 0.0


def _infer_gs_nodes(graph: object) -> Set[str]:
    if not isinstance(graph, nx.Graph):
        return set()
    gs_nodes: Set[str] = set()
    try:
        for node, data in graph.nodes(data=True):
            label = str(node)
            if isinstance(data, dict):
                node_type = str(data.get("type", "")).lower()
                if node_type in {"ground_station", "ground", "gs"}:
                    gs_nodes.add(label)
                    continue
            if label.lower().startswith("gs") or "ground" in label.lower():
                gs_nodes.add(label)
    except Exception:
        return set()
    return gs_nodes


def _collect_attack_nodes(network: LEONetworkModel) -> Set[str]:
    if not network.artifacts:
        return set()
    metrics = network.artifacts.metrics
    offline = metrics.get("offline_satellites", [])
    if isinstance(offline, list):
        return {str(node) for node in offline}
    return set()


def _build_robustness_results(
        network: LEONetworkModel,
        evaluation: Optional[Dict[str, object]],
        baseline_flat: Optional[Dict[str, float]] = None,
        attacked_flat: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, object], Dict[str, object], nx.Graph, Set[str]]:
    graph_obj = None
    if hasattr(network, "_leocraft_graph"):
        graph_obj = network._leocraft_graph()
    attacked_graph = graph_obj if isinstance(graph_obj, nx.Graph) else nx.Graph()
    gs_nodes = _infer_gs_nodes(graph_obj)
    baseline_snapshot = evaluation.get("baseline", {}) if evaluation else {}
    attacked_snapshot = evaluation.get("post_threat", {}) if evaluation else {}

    baseline_result: Dict[str, object] = {
        "pair_stats": {},
        "coverage": {},
        "total_throughput": _extract_total_throughput(baseline_flat),
        "throughput_by_gs": {},
        "graph": graph_obj,
        "gs_nodes": gs_nodes,
    }
    attacked_result: Dict[str, object] = {
        "pair_stats": {},
        "coverage": {},
        "total_throughput": _extract_total_throughput(attacked_flat),
        "throughput_by_gs": {},
        "graph": graph_obj,
        "gs_nodes": gs_nodes,
        "attack_nodes": _collect_attack_nodes(network),
    }

    return baseline_result, attacked_result, attacked_graph, gs_nodes


def main() -> None:
    args = parse_args()
    log_status("解析命令行参数完成")
    output_dir = args.output.resolve()
    args.output = output_dir
    log_status(f"输出目录规范化: {output_dir}")
    evaluator = SimpleImpactEvaluator()
    generator = LLMScenarioGenerator()
    agents = [
        SatelliteDamageAgent(evaluator, generator),
        CongestionCollapseAgent(evaluator, generator),
        ProtocolAttackAgent(evaluator, generator),
    ]
    manager = MultiAgentManager(agents)
    context = build_context()
    log_status("已构建默认的仿真上下文，开始运行多智能体生成威胁场景")

    best_agent, best_payload, run_stats = manager.run(context)
    log_status(f"多智能体完成评分，选中代理: {best_agent.name}, 场景: {best_agent.scenario.name}")

    repo = ScenarioRepository(output_dir)
    scenario_path = repo.save(best_agent.scenario, best_payload)
    log_status(f"威胁场景JSON已保存: {scenario_path}")

    # 从 multi-agent 生成的 JSON 文件中读取威胁场景，并构建威胁模型实例
    try:
        threat_configs = load_threat_json(scenario_path)
    except ValueError as exc:
        log_status(f"威胁配置解析失败，将跳过物理威胁模型注入: {exc}")
        threat_configs = []
    threat_models = build_threat_models(threat_configs) if threat_configs else []
    if threat_models:
        log_status(
            f"已加载威胁模型 {len(threat_models)} 个: {[cfg['type'] for cfg in threat_configs]}"
        )
    else:
        log_status("未加载额外威胁模型，继续执行默认脚本")

    codegen = ThreatCodegenAgent(output_dir / "generated_code")
    script_path, executor = codegen.build_script(best_agent.scenario, best_payload)
    log_status(f"威胁脚本已生成: {script_path}")

    leocraft_output = output_dir / "leocraft_starlink"
    network = LEONetworkModel(
        args.network_name,
        context,
        leocraft_output=leocraft_output,
    )
    log_status("初始化LEO网络模型完成，准备执行威胁脚本注入网络")
    executor(network)
    log_status("威胁脚本执行完毕，开始评估网络性能变化")

    # === Threat model execution loop ===
    sim_state = network  # sim_state encapsulates the constellation, metrics, and topology
    num_steps = int(best_payload.get("duration_steps") or best_payload.get("duration") or 1)
    for t in range(num_steps):
        # 1) 调用所有威胁模型，更新 sim_state（包含节点/链路）
        for model in threat_models:
            model.update(sim_state, t)

        # 2) （可选）路由和性能计算，若可用则在每步刷新指标
        if network.artifacts:
            network.artifacts.recompute_performance()
    evaluation = network.evaluate_performance_metrics()
    network_path = output_dir / "network_timeline.json"
    network.save(network_path)
    log_status(f"网络时间线与性能快照已保存: {network_path}")

    baseline_snapshot = evaluation.get("baseline", {}) if evaluation else {}
    attacked_snapshot = evaluation.get("post_threat", {}) if evaluation else {}
    baseline_flat = flatten_performance_snapshot(baseline_snapshot)
    attacked_flat = flatten_performance_snapshot(attacked_snapshot)

    baseline_result, attacked_result, attacked_graph, gs_nodes = _build_robustness_results(
        network, evaluation, baseline_flat, attacked_flat
    )
    region_of_gs_mapping: Dict[str, str] = {}  # TODO: populate with real region mapping when available
    robustness_evaluator = RobustnessEvaluator(
        baseline_stats=baseline_result.get("pair_stats", {}),
        attacked_stats=attacked_result.get("pair_stats", {}),
        attacked_graph=attacked_graph,
        gs_nodes=gs_nodes,
        baseline_coverage=baseline_result.get("coverage", {}),
        attacked_coverage=attacked_result.get("coverage", {}),
        baseline_total_throughput=float(baseline_result.get("total_throughput", 0.0)),
        attacked_total_throughput=float(attacked_result.get("total_throughput", 0.0)),
        baseline_throughput_by_gs=baseline_result.get("throughput_by_gs", {}),
        attacked_throughput_by_gs=attacked_result.get("throughput_by_gs", {}),
        region_of_gs=region_of_gs_mapping,
        attack_nodes=set(attacked_result.get("attack_nodes", set())),
        baseline_result=baseline_result,
        attacked_result=attacked_result,
        graph_for_apv=attacked_graph,
    )
    log_status("开始计算鲁棒性指标")
    robustness_metrics = robustness_evaluator.compute_all()
    log_status(
        "鲁棒性指标计算完成: "
        f"SRR={robustness_metrics.srr:.3f}, DPR={robustness_metrics.dpr:.3f}, "
        f"LCCR={robustness_metrics.lccr:.3f}, CR={robustness_metrics.cr:.3f}, "
        f"TR={robustness_metrics.tr:.3f}, ΔT={robustness_metrics.d_t:.3f}, "
        f"SI={robustness_metrics.si:.3f}, HI={robustness_metrics.hi:.3f}, "
        f"RF={'N/A' if robustness_metrics.rf is None else f'{robustness_metrics.rf:.3f}'}, "
        f"APV_LEO={'N/A' if robustness_metrics.apv_leo is None else f'{robustness_metrics.apv_leo:.3f}'}"
    )
    recorder = EvaluationRecorder(output_dir)
    agent_metrics = compute_agent_metrics(run_stats)
    performance_delta = summarize_delta(
        evaluation.get("delta") if evaluation else None  # type: ignore[arg-type]
    )
    baseline_metrics = compute_performance_metrics(
        evaluation,
        network.artifacts.performance_baseline if network.artifacts else None,
        baseline_flat=baseline_flat,
        post_flat=attacked_flat,
    )
    log_status("多智能体与性能评估指标整理完成，准备写入CSV")
    row = EvaluationRow(
        network=args.network_name,
        selected_agent=best_agent.name,
        scenario=best_agent.scenario.name,
        category=getattr(best_agent.scenario, "category", "uncategorized"),
        score=float(best_payload.get("score", 0.0)),
        agent_metrics=agent_metrics,
        threat_parameters=best_agent.scenario.key_parameters(best_payload),
        performance_delta=performance_delta,
        performance_metrics=baseline_metrics,
        scenario_path=scenario_path,
        script_path=script_path,
        leocraft_export=network.artifacts.export_directory if network.artifacts else None,
    )
    evaluation_csv = recorder.append(row)
    log_status(f"评估摘要CSV写入完成: {evaluation_csv}")
    results_path = output_dir / "experiment_results.csv"
    attack_type = best_payload.get("category", getattr(best_agent.scenario, "category", ""))
    attack_level = best_payload.get("criticality") or best_payload.get("severity") or 1.0
    metrics_row = {
        "attack_type": attack_type,
        "attack_level": attack_level,
        "srr": robustness_metrics.srr,
        "dpr": robustness_metrics.dpr,
        "lccr": robustness_metrics.lccr,
        "cr": robustness_metrics.cr,
        "tr": robustness_metrics.tr,
        "d_t": robustness_metrics.d_t,
        "si": robustness_metrics.si,
        "hi": robustness_metrics.hi,
        "rf": robustness_metrics.rf,
        "apv_leo": robustness_metrics.apv_leo,
    }
    metrics_df = pd.DataFrame([metrics_row])
    write_header = not results_path.exists()
    metrics_df.to_csv(results_path, index=False, mode="a" if not write_header else "w", header=write_header)
    log_status(f"鲁棒性实验结果已写入: {results_path}")

    print("=== Multi-Agent Risk Scenario Execution ===")
    print(f"Selected Agent: {best_agent.name}")
    print(f"Scenario: {best_agent.scenario.name}")
    print(f"Score: {best_payload['score']:.2f}")
    print("Saved Scenario Payload:", scenario_path)
    print("Generated Threat Script:", script_path)
    print("Saved Network Timeline:", network_path)
    if network.artifacts and network.artifacts.export_directory:
        print("LEOCraft Exports:", network.artifacts.export_directory)
    print("Robustness Metrics CSV:", results_path)
    if evaluation:
        delta = evaluation.get("delta", {})
        headline = ", ".join(
            f"{key}Δ={value:+.3f}" for key, value in list(delta.items())[:5]
        )
        if not headline:
            headline = "无显著的LEOCraft性能变化"
        print("Performance Delta:", headline)

    print("Evaluation Summary CSV:", evaluation_csv)
    if not evaluation_csv.exists():
        raise FileNotFoundError(
            f"评估摘要文件未生成，期望路径: {evaluation_csv}. 请检查输出目录是否可写或是否存在提前退出。"
        )

    print("\n--- Execution Steps ---")
    print("1. 构建LEO星地网络上下文 (ScenarioContext)。")
    print("2. 多智能体独立生成威胁场景并打分，按类别存储为JSON。")
    print("3. 代码生成Agent读取JSON参数生成可执行威胁脚本。")
    print("4. 通过生成的Python脚本将威胁模型注入LEO网络模型并记录扰动日志。")
    print("5. 输出保存路径、脚本路径与性能变化以便后续分析。")


if __name__ == "__main__":
    main()
