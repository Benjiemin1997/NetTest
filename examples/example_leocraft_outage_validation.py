"""Starlink-style LEOCraft outage validation without threat agents.

This example mirrors :mod:`examples.example_starlink` by building the same
Starlink constellation via LEOCraft, then simulates a 30-minute outage where
50% of satellites are unavailable. It recomputes throughput before/after the
outage and optionally renders a simple visualization of the drop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

# Ensure repository root is on sys.path so Threat_Define can be imported when
# running directly from the examples folder.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Threat_Define.simulation.environment import LEONetworkModel
from Threat_Define.simulation.leocraft_starlink import flatten_performance_snapshot
from Threat_Define.threat_scenarios.base import ScenarioContext


def _aggregate_throughput(snapshot: Dict[str, object]) -> float:
    """Sum all throughput-related metrics in a performance snapshot."""

    flat = flatten_performance_snapshot(snapshot or {})
    return sum(value for key, value in flat.items() if "throughput" in key.lower())


def run_validation(output_dir: Path, *, verbose: bool = True) -> Dict[str, object]:
    """Build the LEO network, drop 50% satellites for 30 minutes, and compare throughput."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fallback context numbers are only used if LEOCraft summaries are absent.
    context = ScenarioContext(
        satellite_count=3888,
        inter_satellite_links=6000,
        ground_stations=100,
        critical_services=["navigation", "earth-observation", "broadband"],
    )

    network = LEONetworkModel(
        name="LEOCraft-outage-validation",
        context=context,
        leocraft_output=output_dir,
    )

    if network.artifacts is None:
        raise RuntimeError("LEOCraft artifacts unavailable; cannot validate outage impact.")

    baseline_snapshot = network.artifacts.performance_baseline
    baseline_throughput = _aggregate_throughput(baseline_snapshot)

    summary = network.artifacts.summary
    satellite_total = int(summary.get("satellites", context.satellite_count))
    offline_target = max(1, int(satellite_total * 0.5))

    outage = network.disable_satellites(
        offline_target,
        reason="validation: 50% satellites offline for 30 minutes",
    )
    network.inject_disturbance(
        "Validation outage applied",
        {
            "duration_min": 30,
            "offline_nodes": outage.get("offline_nodes", []),
            "offline_count": outage.get("removed", 0),
        },
    )

    performance = network.evaluate_performance_metrics() or {}
    post_snapshot = performance.get("post_threat", {}) or network.artifacts.snapshot_performance()
    post_throughput = _aggregate_throughput(post_snapshot)

    delta = post_throughput - baseline_throughput

    if verbose:
        print(f"Baseline throughput: {baseline_throughput:.3f}")
        print(f"Post-outage throughput: {post_throughput:.3f}")
        print(f"Delta: {delta:.3f}")

    return {
        "baseline_snapshot": baseline_snapshot,
        "post_snapshot": post_snapshot,
        "baseline_throughput": baseline_throughput,
        "post_throughput": post_throughput,
        "delta": delta,
        "offline_nodes": outage.get("offline_nodes", []),
    }


def plot_throughput(baseline: float, post: float, output_dir: Path) -> Path:
    """Render a bar chart comparing throughput before and after the outage."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["baseline", "post-outage"], [baseline, post], color=["#4CAF50", "#E53935"])
    ax.set_ylabel("Throughput (aggregated units)")
    ax.set_title("Throughput impact of 50% satellite outage (30 min)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for idx, val in enumerate([baseline, post]):
        ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom")
    output_path = output_dir / "throughput_outage_validation.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate LEOCraft throughput impact when 50% satellites go offline for 30 minutes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Threat_Define/outputs/leocraft_outage_validation"),
        help="Directory to store LEOCraft exports and visualization",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating a throughput comparison plot",
    )
    args = parser.parse_args()

    results = run_validation(args.output_dir)
    if not args.no_plot:
        plot_path = plot_throughput(
            results["baseline_throughput"],
            results["post_throughput"],
            args.output_dir,
        )
        print(f"Visualization saved to: {plot_path}")


if __name__ == "__main__":
    main()
