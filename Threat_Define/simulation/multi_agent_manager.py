"""Coordinator that runs multiple risk agents to find the highest impact scenario."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from Threat_Define.agents import RiskAgent
from Threat_Define.threat_scenarios import ScenarioContext
from statistics import mean, pstdev


@dataclass
class MultiAgentManager:
    agents: Iterable[RiskAgent]

    def run(
            self,
            context: ScenarioContext,
            *_,
            score_callback: Optional[
                Callable[[RiskAgent, Dict[str, object]], Tuple[float, Dict[str, object]]]
            ] = None,
            **kwargs,
    ) -> Tuple[RiskAgent, Dict[str, object], Dict[str, object]]:
        scores: List[Tuple[float, RiskAgent, Dict[str, object]]] = []
        reports: List[Dict[str, object]] = []
        # Swallow any unexpected keyword arguments for backward compatibility so
        # callers can pass callback-style scoring without raising errors.
        if kwargs:
            print(
                f"[STATUS] MultiAgentManager.run() 收到额外参数 {list(kwargs.keys())}，已忽略"
            )
        for agent in self.agents:
            print(f"[STATUS] 开始执行代理: {agent.name}，场景: {agent.scenario.name}")
            payload = agent.perceive(context)
            print(f"[STATUS] 代理 {agent.name} 完成场景生成，开始评分")
            score_details: Dict[str, object] = {}
            if score_callback:
                try:
                    score, score_details = score_callback(agent, payload)
                except Exception as exc:
                    print(f"[STATUS] 代理 {agent.name} 闭环评分失败，回退启发式: {exc}")
                    score = agent.evaluate(payload)
            else:
                score = agent.evaluate(payload)
            payload["score_details"] = score_details
            print(f"[STATUS] 代理 {agent.name} 评分完成，得分: {score:.3f}")
            scores.append((score, agent, payload))
            reports.append(
                {
                    "agent": agent.name,
                    "scenario": agent.scenario.name,
                    "category": getattr(agent.scenario, "category", "uncategorized"),
                    "payload": payload,
                    "score": score,
                }
            )
        scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_agent, best_payload = scores[0]
        best_payload = dict(best_payload)
        best_payload.setdefault("scenario", best_agent.scenario.name)
        best_payload.setdefault("category", getattr(best_agent.scenario, "category", "uncategorized"))
        best_payload["score"] = best_score
        print(
            "[STATUS] 代理选择完成，最高分代理: "
            f"{best_agent.name} (场景: {best_agent.scenario.name}, 得分: {best_score:.3f})"
        )

        score_values = [item[0] for item in scores]
        run_metrics = {
            "agents_considered": len(scores),
            "scores": score_values,
            "avg_score": mean(score_values) if score_values else 0.0,
            "score_std": pstdev(score_values) if len(score_values) > 1 else 0.0,
            "unique_categories": len({r[1].scenario.category for r in scores}),
            "unique_scenarios": len({r[1].scenario.name for r in scores}),
            "reports": reports,
        }

        return best_agent, best_payload, run_metrics
