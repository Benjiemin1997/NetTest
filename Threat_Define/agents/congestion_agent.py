"""Agent that models network congestion collapse scenarios."""
from __future__ import annotations

from typing import Dict

from .base import RiskAgent
from llm_client import LLMScenarioGenerator
from threat_scenarios.base import ScenarioContext
from threat_scenarios.network_congestion import NetworkCongestionScenario



class CongestionCollapseAgent(RiskAgent):

    def __init__(
        self, evaluator, generator: LLMScenarioGenerator | None = None
    ) -> None:
        super().__init__(
            "CongestionCollapseAgent", NetworkCongestionScenario(generator), evaluator
        )

    def perceive(self, context: ScenarioContext) -> Dict[str, object]:
        payload = self.scenario.generate(context)
        heuristic = 1.2 + context.inter_satellite_links / max(100, context.inter_satellite_links * 2)
        payload["criticality"] = max(float(payload.get("criticality", 1.0)), heuristic)
        payload.setdefault(
            "analysis",
            "LLM output highlights crosslink saturation and recommends QoS escalation paths.",
        )
        return payload