
from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Dict, List, Optional




def _jsonify(value: object) -> object:
    """Best-effort conversion of nested structures into JSON-serializable data."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _jsonify(val)
            for key, val in vars(value).items()
            if not key.startswith("__")
        }
    return str(value)


def _capture_component_snapshot(component: object) -> Dict[str, object]:
    """Extract metric-like attributes from a LEOCraft performance object."""

    snapshot: Dict[str, object] = {}
    candidates = ("metrics", "results", "stats", "summary")
    for attr in candidates:
        if not hasattr(component, attr):
            continue
        value = getattr(component, attr)
        if value is None:
            continue
        json_value = _jsonify(value)
        if json_value:
            snapshot[attr] = json_value

    if not snapshot:
        scalars: Dict[str, object] = {}
        for key, value in vars(component).items():
            if key.startswith("__"):
                continue
            if isinstance(value, (str, int, float, bool)):
                scalars[key] = value
        if scalars:
            snapshot["scalars"] = scalars

    return snapshot


def _capture_performance_metrics(
    throughput: object, coverage: object, stretch: object
) -> Dict[str, object]:
    """Create a structured snapshot of the LEOCraft performance results."""

    return {
        "throughput": _capture_component_snapshot(throughput),
        "coverage": _capture_component_snapshot(coverage),
        "stretch": _capture_component_snapshot(stretch),
    }


def _flatten_numeric_metrics(data: object, prefix: str = "") -> Dict[str, float]:
    """Flatten nested metric structures into a mapping of dotted-path -> value."""

    metrics: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(_flatten_numeric_metrics(value, new_prefix))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            new_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            metrics.update(_flatten_numeric_metrics(value, new_prefix))
    elif isinstance(data, Number):
        metrics[prefix or "value"] = float(data)
    return metrics


def compute_metric_delta(
    baseline: Dict[str, object],
    current: Dict[str, object],
) -> Dict[str, float]:
    """Compute numeric deltas between two performance snapshots."""

    base_flat = _flatten_numeric_metrics(baseline)
    current_flat = _flatten_numeric_metrics(current)
    delta: Dict[str, float] = {}
    for key in sorted(set(base_flat.keys()) & set(current_flat.keys())):
        delta[key] = round(current_flat[key] - base_flat[key], 6)
    return delta


def flatten_performance_snapshot(snapshot: Dict[str, object]) -> Dict[str, float]:
    """Expose numeric values contained in a performance snapshot."""

    return _flatten_numeric_metrics(snapshot)

@dataclass
class SimulationArtifacts:
    """Container with references and metadata from a LEOCraft simulation run."""

    constellation: object
    throughput: object
    coverage: object
    stretch: object
    duration_seconds: float
    summary: Dict[str, object]
    metrics: Dict[str, object]
    performance_baseline: Dict[str, object]
    export_directory: Optional[Path] = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "summary": self.summary,
            "duration_minutes": round(self.duration_seconds / 60, 2),
            "metrics": self.metrics,
            "performance_baseline": self.performance_baseline,
        }
        if self.export_directory is not None:
            data["export_directory"] = str(self.export_directory)
        return data

    def snapshot_performance(self) -> Dict[str, object]:
        return _capture_performance_metrics(self.throughput, self.coverage, self.stretch)

    def recompute_performance(self) -> Dict[str, object]:
        """Re-run LEOCraft metrics after the constellation has been perturbed."""

        try:
            regenerate = getattr(self.constellation, "generate_routes", None)
            if callable(regenerate):
                regenerate()
        except Exception:  # pragma: no cover - best effort with optional dependency
            pass

        for component in (self.throughput, self.coverage, self.stretch):
            try:
                rebuild = getattr(component, "build", None)
                recompute = getattr(component, "compute", None)
                if callable(rebuild):
                    rebuild()
                if callable(recompute):
                    recompute()
            except Exception:  # pragma: no cover - optional dependency safety
                continue

        return self.snapshot_performance()



def build_starlink_constellation(
    *, output_dir: Optional[Path] = None, verbose: bool = False
) -> SimulationArtifacts:
    """Reproduce the public Starlink example provided by LEOCraft.

    The helper mirrors the reference script that constructs the Starlink network
    with three shells, runs the FSPL loss model, and computes throughput,
    coverage, and latency/stretch metrics. Results can optionally be exported to
    ``output_dir`` for later analysis.
    """

    import time

    from LEOCraft.attenuation.fspl import FSPL
    from LEOCraft.constellations.LEO_constellation import LEOConstellation
    from LEOCraft.dataset import GroundStationAtCities, InternetTrafficAcrossCities
    from LEOCraft.performance.basic.coverage import Coverage
    from LEOCraft.performance.basic.stretch import Stretch
    from LEOCraft.performance.basic.throughput import Throughput
    from LEOCraft.satellite_topology.plus_grid_shell import PlusGridShell
    from LEOCraft.user_terminals.ground_station import GroundStation

    shell_configs: List[Dict[str, float]] = [
        {
            "id": 0,
            "orbits": 72,
            "sat_per_orbit": 22,
            "altitude_m": 550_000.0,
            "inclination_degree": 53.0,
            "angle_of_elevation_degree": 25.0,
            "phase_offset": 50.0,
        },
        {
            "id": 1,
            "orbits": 72,
            "sat_per_orbit": 22,
            "altitude_m": 540_000.0,
            "inclination_degree": 53.2,
            "angle_of_elevation_degree": 25.0,
            "phase_offset": 50.0,
        },
        {
            "id": 2,
            "orbits": 36,
            "sat_per_orbit": 20,
            "altitude_m": 570_000.0,
            "inclination_degree": 70.0,
            "angle_of_elevation_degree": 25.0,
            "phase_offset": 50.0,
        },
    ]

    loss_parameters = {
        "frequency_hz": 28.5e9,
        "tx_power_dbm": 98.4,
        "bandwidth_hz": 0.5e9,
        "g_over_t": 13.6,
        "tx_antenna_gain_db": 34.5,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    loss_model = FSPL(
        loss_parameters["frequency_hz"],
        loss_parameters["tx_power_dbm"],
        loss_parameters["bandwidth_hz"],
        loss_parameters["g_over_t"],
    )
    loss_model.set_Tx_antenna_gain(gain_dB=loss_parameters["tx_antenna_gain_db"])

    constellation = LEOConstellation("Starlink")
    constellation.v.verbose = verbose
    constellation.add_ground_stations(
        GroundStation("/home/user4/NetTest/dataset/ground_stations/cities_sorted_by_estimated_2025_pop_top_100.csv")
    )

    for config in shell_configs:
        constellation.add_shells(PlusGridShell(**config))

    constellation.set_time()
    constellation.set_loss_model(loss_model)
    constellation.build()
    constellation.create_network_graph()
    constellation.generate_routes()

    throughput = Throughput(
        constellation,
        "/home/user4/NetTest/dataset/traffic_metrics/population_only_tm_Gbps_100.json",
    )
    throughput.build()
    throughput.compute()

    coverage = Coverage(constellation)
    coverage.build()
    coverage.compute()

    stretch = Stretch(constellation)
    stretch.build()
    stretch.compute()

    end_time = time.perf_counter()

    if output_dir is not None:
        constellation.export_gsls(output_dir)
        constellation.export_routes(output_dir)
        constellation.export_no_path_found(output_dir)
        constellation.export_k_path_not_found(output_dir)
        for shell in constellation.shells:
            shell.export_satellites(output_dir)
            shell.export_isls(output_dir)
        constellation.ground_stations.export(output_dir)
        throughput.export_path_selection(output_dir)
        throughput.export_LP_model(output_dir)
        stretch.export_stretch_dataset(output_dir)

    total_satellites = sum(
        int(cfg["orbits"]) * int(cfg["sat_per_orbit"]) for cfg in shell_configs
    )
    performance_baseline = _capture_performance_metrics(
        throughput, coverage, stretch
    )
    summary = {
        "satellites": total_satellites,
        "ground_stations": 100,  # GroundStationAtCities.TOP_100 dataset
        "shells": len(shell_configs),
        "shell_details": shell_configs,
    }

    metrics = {
        "loss_model": {
            "type": "FSPL",
            **loss_parameters,
        },
        "traffic_model": str(InternetTrafficAcrossCities.ONLY_POP_100),
        "throughput_class": throughput.__class__.__name__,
        "coverage_class": coverage.__class__.__name__,
        "stretch_class": stretch.__class__.__name__,
        "performance_baseline": performance_baseline,
    }

    return SimulationArtifacts(
        constellation=constellation,
        throughput=throughput,
        coverage=coverage,
        stretch=stretch,
        duration_seconds=end_time - start_time,
        summary=summary,
        metrics=metrics,
        performance_baseline=performance_baseline,
        export_directory=output_dir,
    )