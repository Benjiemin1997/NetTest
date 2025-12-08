
from typing import Dict

from .base import RiskAgent
from llm_client import LLMScenarioGenerator
from threat_scenarios.base import ScenarioContext
from threat_scenarios.satellite_node_failure import SatelliteNodeFailureScenario


class SatelliteDamageAgent(RiskAgent):
    """Collects observations and proposes catastrophic node failure scenarios."""

    def __init__(
        self, evaluator, generator: LLMScenarioGenerator | None = None
    ) -> None:
        super().__init__(
            "SatelliteDamageAgent", SatelliteNodeFailureScenario(generator), evaluator
        )

    def perceive(self, context: ScenarioContext) -> Dict[str, object]:
        payload = self.scenario.generate(context)
        heuristic = 1.6 if context.satellite_count < 60 else 1.25
        payload["criticality"] = max(float(payload.get("criticality", 1.0)), heuristic)
        payload.setdefault(
            "analysis",
            "LLM-driven assessment prioritises resilience gaps in orbital plane redundancy.",
        )
        return payload