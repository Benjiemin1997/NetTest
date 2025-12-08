"""Scenario modeling cascading congestion and collapse of inter-satellite links."""
from __future__ import annotations

from typing import Dict

from .base import LEONetwork, ScenarioContext, ThreatScenario


class NetworkCongestionScenario(ThreatScenario):
    """Create protocol-level congestion collapse in the LEO mesh."""

    def __init__(self, generator: "LLMScenarioGenerator | None" = None) -> None:
        super().__init__(name="Network Congestion Collapse", category="network_congestion")
        self._generator = generator

    def generate(self, context: ScenarioContext) -> Dict[str, object]:
        def fallback_payload() -> Dict[str, object]:
            congested_links = max(1, context.inter_satellite_links // 5)
            saturation = min(95, 60 + context.ground_stations * 2)
            return {
                "congested_links": congested_links,
                "description": (
                    "Malicious telemetry replay overloads {links} crosslinks leading to"
                    " cascading queue buildups and widespread packet loss."
                ).format(links=congested_links),
                "impact_duration_minutes": 45 + context.ground_stations,
                "mitigation": (
                    "Throttle bulk transfers, enforce QoS preemption, and enable dynamic"
                    " congestion control overrides through SDN policies."
                ),
                "steps": [
                    "Seed spoofed telemetry bursts on polar relays to force rerouting.",
                    "Exploit routing convergence lag to overload secondary interlinks.",
                    "Trigger cascading backpressure that starves ground station traffic.",
                ],
                "criticality": 1.1 + saturation / 200,
            }

        if self._generator:
            schema = {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "congested_links": {"type": "integer"},
                    "impact_duration_minutes": {"type": "number"},
                    "mitigation": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "criticality": {"type": "number"},
                },
                "required": [
                    "description",
                    "congested_links",
                    "impact_duration_minutes",
                    "steps",
                ],
                "additionalProperties": True,
            }
            payload = self._generator.generate_threat(
                topic="network congestion collapse",
                context=context,
                guidance=(
                    "Detail adversarial techniques that create congestion storms and"
                    " list operational recovery tasks."
                ),
                schema=schema,
                fallback=fallback_payload,
            )
        else:
            payload = fallback_payload()

        payload["congested_links"] = int(payload.get("congested_links", 1))
        payload["impact_duration_minutes"] = float(payload.get("impact_duration_minutes", 30))
        payload["criticality"] = float(payload.get("criticality", 1.1))
        steps = payload.get("steps", [])
        if isinstance(steps, str):
            steps = [step.strip() for step in steps.splitlines() if step.strip()]
        payload["steps"] = list(steps)
        payload.setdefault(
            "mitigation",
            "Enable emergency QoS enforcement and shift critical services to unaffected relays.",
        )
        payload.setdefault("scenario", self.name)
        return payload

    def apply(self, network: LEONetwork, payload: Dict[str, object]) -> None:
        result = {}
        if hasattr(network, "congest_links"):
            try:
                result = network.congest_links(
                    int(payload["congested_links"]),
                    reason=payload.get("description", "congestion collapse"),
                )
            except Exception as exc:  # pragma: no cover - optional dependency guard
                network.log_event("注入链路拥塞时出现异常: " + str(exc))
        network.log_event(
            "Injecting congestion collapse impacting {links} inter-satellite links.".format(
                links=payload["congested_links"]
            )
        )
        network.inject_disturbance(
            description="Inter-satellite link congestion",
            impact={
                "links": payload["congested_links"],
                "duration_min": payload["impact_duration_minutes"],
                "network_effect": result,
            },
        )

    def key_parameters(self, payload: Dict[str, object]) -> Dict[str, object]:
        return {
            "congested_links": int(payload.get("congested_links", 0)),
            "impact_duration_minutes": float(payload.get("impact_duration_minutes", 0)),
            "criticality": float(payload.get("criticality", 0)),
        }
