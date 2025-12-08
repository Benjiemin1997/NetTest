"""Facilities for persisting generated scenarios to disk."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from threat_scenarios.base import ThreatScenario


@dataclass
class ScenarioRepository:
    """Persist scenarios as JSON documents for traceability and auditing."""

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, scenario: ThreatScenario | str, payload: Dict[str, object]) -> Path:
        """Persist a scenario payload into a category-specific JSON file.

        Accepts either a ``ThreatScenario`` instance or a plain string name to
        remain tolerant of legacy callers that pass ``scenario.name`` directly.
        In the latter case, the category falls back to the payload metadata or
        ``"uncategorized"`` if no hint is available.
        """

        if isinstance(scenario, ThreatScenario):
            scenario_name = scenario.name
            scenario_category = scenario.category
            key_parameters = scenario.key_parameters(payload)
        else:
            scenario_name = str(scenario)
            scenario_category = (
                    payload.get("category")
                    or payload.get("scenario_category")
                    or payload.get("threat_category")
                    or "uncategorized"
            )
            key_params_raw = payload.get("key_parameters", {})
            key_parameters = key_params_raw if isinstance(key_params_raw, dict) else {}

        serializable = {
            "scenario": scenario_name,
            "category": scenario_category,
            "payload": payload,
            "key_parameters": key_parameters,
        }
        category_dir = self.output_dir / scenario_category
        category_dir.mkdir(parents=True, exist_ok=True)
        path = category_dir / f"{scenario_name.replace(' ', '_').lower()}.json"
        path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        return path