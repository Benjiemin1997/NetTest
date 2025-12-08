"""Agent that proposes protocol layer exploit scenarios."""
from __future__ import annotations

from typing import Dict

from llm_client import LLMScenarioGenerator
from threat_scenarios.base import ScenarioContext
from threat_scenarios.protocol_attack import ProtocolAttackScenario

from Threat_Define.agents import RiskAgent


class ProtocolAttackAgent(RiskAgent):
    """Generates protocol exploitation threats focusing on control channels."""

    def __init__(
        self, evaluator, generator: LLMScenarioGenerator | None = None
    ) -> None:
        super().__init__(
            "ProtocolAttackAgent", ProtocolAttackScenario(generator), evaluator
        )

    def perceive(self, context: ScenarioContext) -> Dict[str, object]:
        payload = self.scenario.generate(context)
        heuristic = 1.45 if "navigation" in context.critical_services else 1.15
        payload["criticality"] = max(float(payload.get("criticality", 1.0)), heuristic)
        payload.setdefault(
            "analysis",
            "LLM reasoning emphasises control-plane trust chains and response latency.",
        )
        return payload