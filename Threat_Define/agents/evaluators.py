"""Evaluators responsible for scoring threat payloads."""
from __future__ import annotations

from typing import Dict

from Threat_Define.threat_scenarios import ThreatScenario


class SimpleImpactEvaluator:
    """Heuristic evaluator that approximates risk severity using payload metadata."""

    def score(self, scenario: ThreatScenario, payload: Dict[str, object]) -> float:
        score = 1.0
        if "damaged_nodes" in payload:
            score += payload["damaged_nodes"] * 1.5
        if "congested_links" in payload:
            score += payload["congested_links"] * 1.2
        if "impact_duration_minutes" in payload:
            score += payload["impact_duration_minutes"] / 10
        if "exploited_nodes" in payload:
            score += payload["exploited_nodes"] * 2
        if "criticality" in payload:
            score *= payload["criticality"]
        return float(score)