"""Scenario describing an adversary targeting protocol layers of the LEO network."""
from __future__ import annotations

from typing import Dict

from .base import LEONetwork, ScenarioContext, ThreatScenario

class ProtocolAttackScenario(ThreatScenario):
    """Craft an attack exploiting protocol vulnerabilities and control channels."""

    def __init__(self, generator: "LLMScenarioGenerator | None" = None) -> None:
        super().__init__(name="Protocol Level Attack", category="protocol_attack")
        self._generator = generator

    def generate(self, context: ScenarioContext) -> Dict[str, object]:
        def fallback_payload() -> Dict[str, object]:
            exploited_nodes = min(6, max(2, context.satellite_count // 15))
            return {
                "attack_vector": "Malformed handover negotiation packets",
                "target_protocol": "LEO Inter-satellite Routing Protocol (LIRP)",
                "description": (
                    "Adversary injects malformed control frames to desynchronise state,"
                    " forcing repeated failovers and watchdog resets across constellation nodes."
                ),
                "exploited_nodes": exploited_nodes,
                "impact_duration_minutes": 60,
                "steps": [
                    "Harvest firmware images from development uplink for protocol insights.",
                    "Craft poisoned handover frames targeting roaming satellites.",
                    "Exploit reset storms to open maintenance channels for persistence.",
                ],
                "criticality": 1.6,
                "mitigation": (
                    "Deploy signed control messages, isolate compromised nodes, rotate keys,"
                    " and patch vulnerable firmware images."
                ),
            }

        if self._generator:
            schema = {
                "type": "object",
                "properties": {
                    "attack_vector": {"type": "string"},
                    "target_protocol": {"type": "string"},
                    "description": {"type": "string"},
                    "exploited_nodes": {"type": "integer"},
                    "impact_duration_minutes": {"type": "number"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "criticality": {"type": "number"},
                    "mitigation": {"type": "string"},
                },
                "required": [
                    "attack_vector",
                    "target_protocol",
                    "description",
                    "exploited_nodes",
                ],
                "additionalProperties": True,
            }
            payload = self._generator.generate_threat(
                topic="protocol level cyber attack",
                context=context,
                guidance=(
                    "Describe how protocol weaknesses can be weaponized and outline both"
                    " exploitation steps and containment strategies."
                ),
                schema=schema,
                fallback=fallback_payload,
            )
        else:
            payload = fallback_payload()

        payload.setdefault("attack_vector", "Unknown control plane exploit")
        payload.setdefault("target_protocol", "Inter-satellite routing protocol")
        payload["exploited_nodes"] = int(payload.get("exploited_nodes", 2))
        payload["impact_duration_minutes"] = float(payload.get("impact_duration_minutes", 50))
        payload["criticality"] = float(payload.get("criticality", 1.5))
        steps = payload.get("steps", [])
        if isinstance(steps, str):
            steps = [step.strip() for step in steps.splitlines() if step.strip()]
        payload["steps"] = list(steps)
        payload.setdefault("mitigation", "Rotate credentials and harden control-plane parsing logic.")
        payload.setdefault("scenario", self.name)
        return payload

    def apply(self, network: LEONetwork, payload: Dict[str, object]) -> None:
        result = {}
        if hasattr(network, "compromise_protocol"):
            try:
                result = network.compromise_protocol(
                    int(payload["exploited_nodes"]),
                    vector=payload.get("attack_vector", "unknown"),
                    duration=float(payload.get("impact_duration_minutes", 0.0)),
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                network.log_event("注入协议攻击时出现异常: " + str(exc))
        network.log_event(
            "Executing protocol attack on {nodes} satellites via {vector}.".format(
                nodes=payload["exploited_nodes"], vector=payload["attack_vector"]
            )
        )
        network.inject_disturbance(
            description="Protocol level exploit",
            impact={
                "attack_vector": payload["attack_vector"],
                "nodes": payload["exploited_nodes"],
                "network_effect": result,
            },
        )

    def key_parameters(self, payload: Dict[str, object]) -> Dict[str, object]:
        return {
            "exploited_nodes": int(payload.get("exploited_nodes", 0)),
            "impact_duration_minutes": float(payload.get("impact_duration_minutes", 0)),
            "attack_vector": payload.get("attack_vector", ""),
            "criticality": float(payload.get("criticality", 0)),
        }