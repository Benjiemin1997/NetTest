"""Evaluation helpers for measuring multi-agent generation and network impact."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

from .leocraft_starlink import flatten_performance_snapshot



def _aggregate_metric(flat_snapshot: Dict[str, float] | None, keyword: str, *, mode: str = "mean") -> float:
    values = [value for key, value in (flat_snapshot or {}).items() if keyword in key.lower()]

    if not values:
        return 0.0
    if mode == "sum":
        return float(sum(values))
    return float(mean(values))


def summarize_delta(delta: Dict[str, float] | None) -> Dict[str, float]:
    if not delta:
        return {
            "delta_abs_sum": 0.0,
            "delta_max_abs": 0.0,
            "delta_throughput": 0.0,
            "delta_coverage": 0.0,
            "delta_stretch": 0.0,
        }

    abs_values = [abs(value) for value in delta.values()]
    throughput_delta = sum(value for key, value in delta.items() if "throughput" in key.lower())
    coverage_delta = sum(value for key, value in delta.items() if "coverage" in key.lower())
    stretch_delta = sum(value for key, value in delta.items() if "stretch" in key.lower())

    return {
        "delta_abs_sum": float(sum(abs_values)),
        "delta_max_abs": float(max(abs_values) if abs_values else 0.0),
        "delta_throughput": float(throughput_delta),
        "delta_coverage": float(coverage_delta),
        "delta_stretch": float(stretch_delta),
    }


def compute_agent_metrics(run_stats: Dict[str, object]) -> Dict[str, object]:
    scores: List[float] = [float(score) for score in run_stats.get("scores", [])]
    reports: Iterable[Dict[str, object]] = run_stats.get("reports", [])  # type: ignore[assignment]
    categories = {str(report.get("category", "")) for report in reports}
    scenarios = {str(report.get("scenario", "")) for report in reports}

    return {
        "agents_considered": run_stats.get("agents_considered", len(scores)),
        "avg_score": run_stats.get("avg_score", mean(scores) if scores else 0.0),
        "score_std": run_stats.get("score_std", 0.0),
        "unique_categories": run_stats.get("unique_categories", len(categories)),
        "unique_scenarios": run_stats.get("unique_scenarios", len(scenarios)),
        "max_score": max(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
    }


def compute_performance_metrics(
    performance: Optional[Dict[str, object]],
    baseline_snapshot: Optional[Dict[str, object]],
    *,
    baseline_flat: Optional[Dict[str, float]] = None,
    post_flat: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if not performance:
        return {
            "throughput_baseline": 0.0,
            "coverage_baseline": 0.0,
            "stretch_baseline": 0.0,
        }

    post_snapshot = performance.get("post_threat", {})
    baseline_flat = baseline_flat or flatten_performance_snapshot(baseline_snapshot or {})
    post_flat = post_flat or flatten_performance_snapshot(post_snapshot)
    return {
        # Use sums for throughput to align with network_timeline totals; other metrics
        # remain averaged to avoid double-counting coverage entries.
        "throughput_baseline": _aggregate_metric(baseline_flat, "throughput", mode="sum"),
        "coverage_baseline": _aggregate_metric(baseline_flat, "coverage"),
        "stretch_baseline": _aggregate_metric(baseline_flat, "stretch"),
        "throughput_post": _aggregate_metric(post_flat, "throughput", mode="sum"),
        "coverage_post": _aggregate_metric(post_flat, "coverage"),
        "stretch_post": _aggregate_metric(post_flat, "stretch"),
    }


@dataclass
class EvaluationRow:
    network: str
    selected_agent: str
    scenario: str
    category: str
    score: float
    agent_metrics: Dict[str, object]
    threat_parameters: Dict[str, object]
    performance_delta: Dict[str, float]
    performance_metrics: Dict[str, float]
    scenario_path: Path
    script_path: Path
    leocraft_export: Optional[Path]

    def to_dict(self) -> Dict[str, object]:
        base = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "network": self.network,
            "selected_agent": self.selected_agent,
            "scenario": self.scenario,
            "category": self.category,
            "score": round(self.score, 3),
            "scenario_path": str(self.scenario_path),
            "script_path": str(self.script_path),
            "leocraft_export": str(self.leocraft_export) if self.leocraft_export else "",
        }

        threat = {f"threat_{key}": value for key, value in self.threat_parameters.items()}
        agent = {f"agent_{key}": value for key, value in self.agent_metrics.items()}
        delta = {f"delta_{key}": value for key, value in self.performance_delta.items()}
        perf = {f"perf_{key}": value for key, value in self.performance_metrics.items()}
        base.update(threat)
        base.update(agent)
        base.update(delta)
        base.update(perf)
        return base


class EvaluationRecorder:
    """Persist evaluation summaries into a CSV file for easy comparison."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "evaluation_summary.csv"

    def append(self, row: EvaluationRow) -> Path:
        row_dict = row.to_dict()
        fieldnames = list(row_dict.keys())
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row_dict)
        print(f"[STATUS] 评估摘要已追加: {self.csv_path}")
        return self.csv_path