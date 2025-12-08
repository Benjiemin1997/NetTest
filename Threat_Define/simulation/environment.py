"""Simple LEO network model used for injecting simulated threat payloads."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from threat_scenarios.base import ScenarioContext

from .leocraft_starlink import (
    SimulationArtifacts,
    build_starlink_constellation,
    compute_metric_delta,
    flatten_performance_snapshot,
)


class LEOCraftIntegrationError(RuntimeError):
    """Raised when the LEOCraft library cannot be used for network construction."""


def _build_with_leocraft(output_dir: Optional[Path]) -> SimulationArtifacts:
    """Construct the Starlink constellation using LEOCraft's public example."""

    try:
        return build_starlink_constellation(output_dir=output_dir, verbose=False)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional pkg
        missing = exc.name or "LEOCraft"
        raise LEOCraftIntegrationError(
            f"缺少 LEOCraft 依赖包 `{missing}`，无法构建真实星座。"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise LEOCraftIntegrationError(
            "调用 LEOCraft 构建 Starlink 星座失败: " + str(exc)
        ) from exc

@dataclass
class NetworkLogEntry:
    event: str
    impact: Dict[str, object]


@dataclass
class LEONetworkModel:
    """In-memory representation of a LEO satellite-ground network."""

    name: str
    context: ScenarioContext
    log: List[NetworkLogEntry] = field(default_factory=list)
    topology: Optional[object] = None
    leocraft_output: Optional[Path] = None
    artifacts: Optional[SimulationArtifacts] = None

    def __post_init__(self) -> None:
        if self.topology is not None:
            return

        try:
            self.artifacts = _build_with_leocraft(self.leocraft_output)
            self.topology = self.artifacts.constellation
            summary = self.artifacts.summary
            self.log_event("LEOCraft: 已构建Starlink星地网络拓扑并计算性能指标。")
            self.log_event(
                "星座详情: 卫星{satellites}颗, 地面站{ground_stations}个, 壳层{shells}个。".format(
                    satellites=summary.get("satellites", "?"),
                    ground_stations=summary.get("ground_stations", "?"),
                    shells=summary.get("shells", "?"),
                )
            )
            baseline = self.artifacts.performance_baseline
            baseline_text = self._format_performance_snapshot(baseline)
            if baseline_text:
                self.log_event("LEOCraft性能基线: " + baseline_text)
            if self.artifacts.export_directory is not None:
                self.log_event(
                    "LEOCraft仿真数据已导出至: "
                    + str(self.artifacts.export_directory)
                )
        except LEOCraftIntegrationError as exc:
            self.log_event(
                "LEOCraft未可用，回退到内置的最小网络模型: " + str(exc)
            )

    def inject_disturbance(self, description: str, impact: Dict[str, object]) -> None:
        self.log.append(NetworkLogEntry(event=description, impact=impact))

    def _leocraft_graph(self):
        if not self.artifacts:
            return None
        constellation = getattr(self.artifacts, "constellation", None)
        if constellation is None:
            return None
        graph = getattr(constellation, "graph", None)
        if graph is None:
            graph = getattr(constellation, "network_graph", None)
        return graph

    def _append_no_path_records(
        self, offline_nodes: List[str], *, reason: str
    ) -> Optional[Path]:
        if not offline_nodes or not self.artifacts:
            return None
        export_dir = self.artifacts.export_directory
        if export_dir is None:
            return None
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / "Starlink_no_path_found.txt"
        lines: List[str] = []
        if path.exists():
            existing = path.read_text(encoding="utf-8").rstrip("\n")
            if existing:
                lines.append(existing)
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        lines.append(f"# threat-injection {timestamp} {reason}")
        for node in offline_nodes:
            lines.append(f"{node}\tUNREACHABLE\t{reason}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def disable_satellites(self, count: int, *, reason: str) -> Dict[str, object]:
        """Remove satellites from the constellation graph to mimic hard failures."""

        if count <= 0:
            return {"removed": 0, "offline_nodes": []}

        offline_nodes: List[str] = []
        removed = 0
        graph = self._leocraft_graph()
        if graph is not None and hasattr(graph, "nodes"):
            try:
                nodes = list(graph.nodes)  # type: ignore[attr-defined]
                nodes = sorted(nodes, key=lambda node: str(node))
                target = nodes[:count]
                if target:
                    removed = len(target)
                    offline_nodes = [str(node) for node in target]
                    if hasattr(graph, "remove_nodes_from"):
                        graph.remove_nodes_from(target)  # type: ignore[attr-defined]
                    else:
                        for node in target:
                            graph.remove_node(node)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - best effort when LEOCraft present
                self.log_event(
                    "无法直接从LEOCraft网络图中删除卫星节点: " + str(exc)
                )
                removed = 0
                offline_nodes = []

        if removed == 0:
            baseline = int(
                self.artifacts.summary.get("satellites", self.context.satellite_count)
            ) if self.artifacts else self.context.satellite_count
            removed = min(count, baseline)
            offline_nodes = [f"SAT-{index:04d}" for index in range(1, removed + 1)]

        if self.artifacts:
            summary = self.artifacts.summary
            baseline_total = int(summary.get("satellites", self.context.satellite_count))
            summary["satellites"] = max(0, baseline_total - removed)
            summary["satellites_offline"] = summary.get("satellites_offline", 0) + removed
            outage_pct = round((removed / max(1, baseline_total)) * 100, 2)
            metrics = self.artifacts.metrics
            metrics["estimated_satellite_outage_pct"] = outage_pct
            metrics["offline_satellites"] = offline_nodes

        self.context.satellite_count = max(0, self.context.satellite_count - removed)

        exported = self._append_no_path_records(offline_nodes, reason=reason)

        return {
            "removed": removed,
            "offline_nodes": offline_nodes,
            "no_path_file": str(exported) if exported else None,
        }

    def congest_links(self, count: int, *, reason: str) -> Dict[str, object]:
        """Remove edges from the constellation graph to mimic congestion collapse."""

        if count <= 0:
            return {"links_removed": 0, "links": []}

        affected_edges: List[str] = []
        removed = 0
        graph = self._leocraft_graph()
        if graph is not None and hasattr(graph, "edges"):
            try:
                edges = list(graph.edges)  # type: ignore[attr-defined]
                edges = sorted(edges, key=lambda edge: (str(edge[0]), str(edge[1])))
                target = edges[:count]
                if target:
                    removed = len(target)
                    affected_edges = [f"{edge[0]}->{edge[1]}" for edge in target]
                    if hasattr(graph, "remove_edges_from"):
                        graph.remove_edges_from(target)  # type: ignore[attr-defined]
                    else:
                        for edge in target:
                            graph.remove_edge(*edge)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - depends on LEOCraft internals
                self.log_event("无法在LEOCraft图中移除链路: " + str(exc))
                removed = 0
                affected_edges = []

        if removed == 0:
            removed = count
            affected_edges = [f"LINK-{index:04d}" for index in range(1, count + 1)]

        if self.artifacts:
            metrics = self.artifacts.metrics
            metrics["congested_links"] = metrics.get("congested_links", 0) + removed
            metrics["congested_link_ids"] = affected_edges

        exported = None
        if self.artifacts and self.artifacts.export_directory is not None:
            export_dir = self.artifacts.export_directory
            export_dir.mkdir(parents=True, exist_ok=True)
            exported = export_dir / "Starlink_congested_links.txt"
            existing = ""
            if exported.exists():
                existing = exported.read_text(encoding="utf-8").rstrip("\n")
            timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            lines = []
            if existing:
                lines.append(existing)
            lines.append(f"# threat-injection {timestamp} {reason}")
            lines.extend(affected_edges)
            exported.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return {
            "links_removed": removed,
            "links": affected_edges,
            "record": str(exported) if exported else None,
        }

    def compromise_protocol(self, nodes: int, *, vector: str, duration: float) -> Dict[str, object]:
        """Record protocol-level compromise metadata for downstream analysis."""

        if nodes <= 0:
            return {"affected": 0}

        affected = nodes
        if self.artifacts:
            metrics = self.artifacts.metrics
            metrics["protocol_attacks"] = metrics.get("protocol_attacks", 0) + 1
            metrics.setdefault("protocol_compromise", []).append(
                {
                    "vector": vector,
                    "nodes": nodes,
                    "duration_min": duration,
                }
            )
        return {"affected": affected}

    # --- Performance evaluation ------------------------------------------------
    def _format_performance_snapshot(self, snapshot: Dict[str, object]) -> str:
        flat = flatten_performance_snapshot(snapshot)
        if not flat:
            return ""
        interesting: List[str] = []
        for key, value in flat.items():
            lower = key.lower()
            if any(token in lower for token in ("throughput", "coverage", "stretch", "latency")):
                interesting.append(f"{key}={value:.3f}")
        if not interesting:
            sample = list(flat.items())[:3]
            interesting = [f"{key}={value:.3f}" for key, value in sample]
        return ", ".join(interesting)

    def _format_delta(self, delta: Dict[str, float]) -> str:
        if not delta:
            return "无显著变化"
        interesting: List[str] = []
        for key, value in delta.items():
            lower = key.lower()
            if any(token in lower for token in ("throughput", "coverage", "stretch", "latency")):
                interesting.append(f"{key}Δ={value:+.3f}")
        if not interesting:
            sample = list(delta.items())[:3]
            interesting = [f"{key}Δ={value:+.3f}" for key, value in sample]
        return ", ".join(interesting)

    def evaluate_performance_metrics(self) -> Optional[Dict[str, object]]:
        """Re-run LEOCraft metrics to determine whether the threat had impact."""

        if not self.artifacts:
            self.log_event("LEOCraft性能评估已跳过：当前使用内置最小网络模型。")
            return None

        baseline = self.artifacts.performance_baseline
        post = self.artifacts.recompute_performance()
        delta = compute_metric_delta(baseline, post)

        summary = self._format_delta(delta)
        self.log_event("威胁注入后重新评估LEOCraft性能: " + summary)

        self.inject_disturbance(
            description="LEOCraft性能评估",
            impact={"baseline": baseline, "post_threat": post, "delta": delta},
        )

        # Persist evaluation results for downstream consumers.
        leocraft_metrics = self.artifacts.metrics
        leocraft_metrics["performance_after_threat"] = post
        leocraft_metrics["performance_delta"] = delta

        return {"baseline": baseline, "post_threat": post, "delta": delta}

    def log_event(self, event: str) -> None:
        self.log.append(NetworkLogEntry(event=event, impact={}))

    def to_dict(self) -> Dict[str, object]:
        return {
            "network": self.name,
            "topology_source": "leocraft" if self.topology is not None else "fallback",
            "timeline": [
                {"event": entry.event, "impact": entry.impact} for entry in self.log
            ],
            "leocraft": self.artifacts.to_dict() if self.artifacts else None,
        }

    def save(self, path: Path) -> None:
        path.write_text(
            __import__("json").dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )