from __future__ import annotations

from typing import Dict

from .base import LEONetwork, ScenarioContext, ThreatScenario

class SatelliteNodeFailureScenario(ThreatScenario):
    """Simulate physical or logical damage that removes satellites from the mesh."""

    def __init__(self, generator: "LLMScenarioGenerator | None" = None) -> None:
        super().__init__(name="Satellite Node Failure", category="satellite_failure")
        self._generator = generator

    def generate(self, context: ScenarioContext) -> Dict[str, object]:
        def fallback_payload() -> Dict[str, object]:
            damaged_nodes = max(1, context.satellite_count // 12)
            impact_minutes = max(30, context.inter_satellite_links // 4)
            return {
                "damaged_nodes": damaged_nodes,
                "affected_services": context.critical_services[:2],
                "description": (
                    "Sustained anti-satellite barrage removes {damaged} nodes and fractures "
                    "mesh routing stability."
                ).format(damaged=damaged_nodes),
                "mitigation": (
                    "Activate emergency crosslink rerouting, reprioritize bandwidth for"
                    " crewed missions, and schedule rapid replacement launches."
                ),
                "impact_duration_minutes": impact_minutes,
                "steps": [
                    "Initial reconnaissance pinpoints vulnerable satellites on key orbits.",
                    "Coordinated strike disrupts crosslinks and isolates regional clusters.",
                    "Fallback routing saturates remaining nodes, delaying service restoration.",
                ],
                "criticality": 1.4 if context.satellite_count < 80 else 1.2,
            }

        if self._generator:
            schema = {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "damaged_nodes": {"type": "integer"},
                    "affected_services": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "mitigation": {"type": "string"},
                    "impact_duration_minutes": {"type": "number"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "criticality": {"type": "number"},
                },
                "required": [
                    "description",
                    "damaged_nodes",
                    "impact_duration_minutes",
                    "steps",
                ],
                "additionalProperties": True,
            }
            payload = self._generator.generate_threat(
                topic="satellite node failure",
                context=context,
                guidance=(
                    "Emphasize cascading failures from physical loss of satellites and"
                    " describe concrete mitigation checkpoints."
                ),
                schema=schema,
                fallback=fallback_payload,
            )
        else:
            payload = fallback_payload()

        payload["damaged_nodes"] = int(payload.get("damaged_nodes", 1))
        payload.setdefault("affected_services", context.critical_services[:2])
        steps = payload.get("steps", [])
        if isinstance(steps, str):
            steps = [step.strip() for step in steps.splitlines() if step.strip()]
        payload["steps"] = list(steps)
        payload["impact_duration_minutes"] = float(payload.get("impact_duration_minutes", 45))
        payload["criticality"] = float(payload.get("criticality", 1.2))
        payload.setdefault("mitigation", "Deploy spares and rebalance routing tables rapidly.")
        payload.setdefault("scenario", self.name)
        return payload

    def apply(self, network: LEONetwork, payload: Dict[str, object]) -> None:
        result = {}
        if hasattr(network, "disable_satellites"):
            try:
                result = network.disable_satellites(
                    int(payload["damaged_nodes"]),
                    reason=payload.get("description", "satellite node failure"),
                )
            except Exception as exc:  # pragma: no cover - runtime safety with optional deps
                network.log_event("注入卫星失效时出现异常: " + str(exc))
        network.log_event(
            f"Applying satellite node failure affecting {payload['damaged_nodes']} nodes."
        )
        network.inject_disturbance(
            description="Satellite node failure",
            impact={
                "nodes_offline": payload["damaged_nodes"],
                "service_disruption": payload["affected_services"],
                "network_effect": result,
            },
        )

    def key_parameters(self, payload: Dict[str, object]) -> Dict[str, object]:
        return {
            "damaged_nodes": int(payload.get("damaged_nodes", 0)),
            "impact_duration_minutes": float(payload.get("impact_duration_minutes", 0)),
            "affected_services": list(payload.get("affected_services", [])),
            "criticality": float(payload.get("criticality", 0)),
        }