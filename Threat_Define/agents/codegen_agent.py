from __future__ import annotations

import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from threat_scenarios.base import ThreatScenario


@dataclass
class ThreatCodegenAgent:
    """Materialise threat payloads into runnable Python injection scripts."""

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_script(
        self, scenario: ThreatScenario, payload: Dict[str, object]
    ) -> tuple[Path, Callable[[object], None]]:
        """Generate a runnable Python file and return its executor callable."""
        category_dir = self.output_dir / scenario.category
        category_dir.mkdir(parents=True, exist_ok=True)
        script_path = category_dir / f"{scenario.name.replace(' ', '_').lower()}_inject.py"
        script_path.write_text(
            self._render_script(scenario, payload), encoding="utf-8"
        )
        module = runpy.run_path(str(script_path))
        execute = module.get("execute")
        if not callable(execute):
            raise RuntimeError("Generated threat script lacks an execute(network) function")
        return script_path, execute  # type: ignore[return-value]

    def _render_script(self, scenario: ThreatScenario, payload: Dict[str, object]) -> str:
        metadata = json.dumps(payload, ensure_ascii=False, indent=2)
        lines = [
            f"\"\"\"Auto-generated injection script for {scenario.name} ({scenario.category}).",
            "This file was produced from a structured JSON payload and can be executed",
            "via `execute(network)` to apply the threat to an active LEONetworkModel.",
            "\"\"\"",
            "",
            "from __future__ import annotations",
            "from typing import Any",
            "",
            "def execute(network: Any) -> None:",
            "    \"\"\"",
            "    Apply the threat model to the provided network instance.",
            "    The network is expected to expose methods matching the scenario category.",
            "    \"\"\"",
            f"    payload = {metadata}",
            "    network.log_event(",
            "        f\"[代码注入] 执行威胁模型: {payload.get('scenario', '未知场景')}\"",
            "    )",
        ]

        if scenario.category == "satellite_failure":
            lines.extend(
                [
                    "    count = int(payload.get(\"damaged_nodes\", 0))",
                    "    reason = payload.get(\"description\", \"satellite node failure\")",
                    "    if hasattr(network, \"disable_satellites\"):",
                    "        try:",
                    "            network.disable_satellites(count, reason=reason)",
                    "        except Exception as exc:",
                    "            network.log_event(f\"执行卫星失效代码时异常: {exc}\")",
                    "    network.inject_disturbance(",
                    "        description=\"Satellite node failure (generated)\",",
                    "        impact={",
                    "            \"nodes_offline\": count,",
                    "            \"reason\": reason,",
                    "            \"criticality\": payload.get(\"criticality\"),",
                    "        },",
                    "    )",
                ]
            )
        elif scenario.category == "network_congestion":
            lines.extend(
                [
                    "    links = int(payload.get(\"congested_links\", 0))",
                    "    reason = payload.get(\"description\", \"inter-satellite congestion\")",
                    "    if hasattr(network, \"congest_links\"):",
                    "        try:",
                    "            network.congest_links(links, reason=reason)",
                    "        except Exception as exc:",
                    "            network.log_event(f\"执行链路拥塞代码时异常: {exc}\")",
                    "    network.inject_disturbance(",
                    "        description=\"Inter-satellite congestion (generated)\",",
                    "        impact={",
                    "            \"links\": links,",
                    "            \"duration_min\": payload.get(\"impact_duration_minutes\"),",
                    "            \"criticality\": payload.get(\"criticality\"),",
                    "        },",
                    "    )",
                ]
            )
        else:  # protocol_attack and other future categories
            lines.extend(
                [
                    "    nodes = int(payload.get(\"exploited_nodes\", 0))",
                    "    vector = payload.get(\"attack_vector\", \"unknown vector\")",
                    "    duration = float(payload.get(\"impact_duration_minutes\", 0))",
                    "    if hasattr(network, \"compromise_protocol\"):",
                    "        try:",
                    "            network.compromise_protocol(nodes, vector=vector, duration=duration)",
                    "        except Exception as exc:",
                    "            network.log_event(f\"执行协议攻击代码时异常: {exc}\")",
                    "    network.inject_disturbance(",
                    "        description=\"Protocol exploit (generated)\",",
                    "        impact={",
                    "            \"attack_vector\": vector,",
                    "            \"nodes\": nodes,",
                    "            \"duration_min\": duration,",
                    "            \"criticality\": payload.get(\"criticality\"),",
                    "        },",
                    "    )",
                ]
            )

        lines.append("")
        return "\n".join(lines)
