"""Demonstrate a 50% satellite outage over 30 minutes without threat agents.

This helper builds the LEOCraft Starlink constellation once, masks half of the
satellites as offline for a 30-minute window, recomputes throughput, and
visualizes the drop. It avoids the multi-agent pipeline so that network impacts
can be inspected directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from Threat_Define.simulation.environment import LEONetworkModel
from Threat_Define.simulation.leocraft_starlink import flatten_performance_snapshot
from Threat_Define.threat_scenarios.base import ScenarioContext


def _extract_throughput(snapshot: Dict[str, object]) -> float:
    """Aggregate all throughput-related numeric entries from a snapshot."""

    flat = flatten_performance_snapshot(snapshot or {})
    return sum(value for key, value in flat.items() if "throughput" in key.lower())


def run_outage_demo(output_dir: Path, *, verbose: bool = True) -> Dict[str, object]:
    """Construct the network, drop 50% of satellites, and measure throughput loss."""

    output_dir.mkdir(parents=True, exist_ok=True)
    context = ScenarioContext(
        satellite_count=120,  # Only used as a fallback if LEOCraft summary is missing
        inter_satellite_links=180,
        ground_stations=25,
        critical_services=["navigation", "earth-observation", "broadband"],
    )

    network = LEONetworkModel(
        name="LEOCraft-outage-demo",
        context=context,
        leocraft_output=output_dir,
    )

    if network.artifacts is None:
        raise RuntimeError("LEOCraft artifacts unavailable; cannot run outage demo.")

    summary = network.artifacts.summary
    baseline_snapshot = network.artifacts.performance_baseline
    baseline_throughput = _extract_throughput(baseline_snapshot)

    satellite_total = int(summary.get("satellites", context.satellite_count))
    offline_target = max(1, int(satellite_total * 0.5))

    outage = network.disable_satellites(
        offline_target,
        reason="demo outage: 50% satellites offline for 30 minutes",
    )
    network.inject_disturbance(
        "Demo outage applied",
        {
            "duration_min": 30,
            "offline_nodes": outage.get("offline_nodes", []),
            "offline_count": outage.get("removed", 0),
        },
    )

    performance = network.evaluate_performance_metrics() or {}
    post_snapshot = performance.get("post_threat", {}) or network.artifacts.snapshot_performance()
    post_throughput = _extract_throughput(post_snapshot)

    delta = post_throughput - baseline_throughput

    if verbose:
        print("Baseline throughput:", baseline_throughput)
        print("Post-outage throughput:", post_throughput)
        print("Delta:", delta)

    return {
        "baseline_snapshot": baseline_snapshot,
        "post_snapshot": post_snapshot,
        "baseline_throughput": baseline_throughput,
        "post_throughput": post_throughput,
        "delta": delta,
        "offline_nodes": outage.get("offline_nodes", []),
    }


def plot_throughput(baseline: float, post: float, output_dir: Path) -> Path:
    """Render a simple bar chart comparing throughput before/after outage."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["baseline", "post-outage"], [baseline, post], color=["#4CAF50", "#E53935"])
    ax.set_ylabel("Throughput (aggregated units)")
    ax.set_title("Throughput impact of 50% satellite outage (30 min)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for idx, val in enumerate([baseline, post]):
        ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom")
    output_path = output_dir / "throughput_outage_demo.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo 50% outage impact without agents")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Threat_Define/outputs/leocraft_outage_demo"),
        help="Directory to store LEOCraft exports and visualization",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating a throughput comparison plot",
    )
    args = parser.parse_args()

    results = run_outage_demo(args.output_dir)
    if not args.no_plot:
        plot_path = plot_throughput(
            results["baseline_throughput"], results["post_throughput"], args.output_dir
        )
        print(f"Visualization saved to: {plot_path}")


if __name__ == "__main__":
    main()
