from __future__ import annotations
from typing import Any

def execute(network: Any) -> None:
    """
    Apply the threat model to the provided network instance.
    The network is expected to expose methods matching the scenario category.
    """
    payload = {
  "congested_links": 36,
  "description": "Malicious telemetry replay overloads 36 crosslinks leading to cascading queue buildups and widespread packet loss.",
  "impact_duration_minutes": 70.0,
  "mitigation": "Throttle bulk transfers, enforce QoS preemption, and enable dynamic congestion control overrides through SDN policies.",
  "steps": [
    "Seed spoofed telemetry bursts on polar relays to force rerouting.",
    "Exploit routing convergence lag to overload secondary interlinks.",
    "Trigger cascading backpressure that starves ground station traffic."
  ],
  "criticality": 1.7,
  "generated_by": "deterministic",
  "scenario": "Network Congestion Collapse",
  "analysis": "LLM output highlights crosslink saturation and recommends QoS escalation paths.",
  "category": "network_congestion",
  "score": 87.03999999999999
}
    network.log_event(
        f"[代码注入] 执行威胁模型: {payload.get('scenario', '未知场景')}"
    )
    links = int(payload.get("congested_links", 0))
    reason = payload.get("description", "inter-satellite congestion")
    if hasattr(network, "congest_links"):
        try:
            network.congest_links(links, reason=reason)
        except Exception as exc:
            network.log_event(f"执行链路拥塞代码时异常: {exc}")
    network.inject_disturbance(
        description="Inter-satellite congestion (generated)",
        impact={
            "links": links,
            "duration_min": payload.get("impact_duration_minutes"),
            "criticality": payload.get("criticality"),
        },
    )
