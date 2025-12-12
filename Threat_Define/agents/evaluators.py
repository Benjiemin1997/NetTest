"""Evaluators responsible for scoring threat payloads."""
from __future__ import annotations

from typing import Dict

from Threat_Define.threat_scenarios import ThreatScenario


class SimpleImpactEvaluator:
    """Heuristic evaluator that approximates risk severity using payload metadata."""

    def score(self, scenario: ThreatScenario, payload: Dict[str, object]) -> float:
        score = 1.0
        details = payload.get("score_details") if isinstance(payload.get("score_details"), dict) else {}
        perf = details.get("evaluation") if isinstance(details, dict) else None
        if isinstance(perf, dict):
            throughput_loss = float(perf.get("throughput_loss", 0.0))
            coverage_loss = float(perf.get("coverage_loss", 0.0))
            stretch_increase = float(perf.get("stretch_increase", 0.0))
            score += throughput_loss * 0.3 + coverage_loss * 50.0 + stretch_increase * 5.0
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
