"""Solar storm induced satellite outage model and demo integration loop."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass
class SolarStormNodeOutageModel:
    """Monte Carlo based SEU-induced failure model for LEO satellites."""

    S: float
    q_rec: float
    k0: float = 3e-7
    a: float = 4.0
    beta: float = 1.5
    c: float = 0.2
    h0: float = 550.0
    dt: float = 1.0
    start_time: int | None = None
    end_time: int | None = None
    rng: random.Random = field(default_factory=random.Random)
    sat_state: Dict[str, int] = field(default_factory=dict)
    _disabled_edges: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    def f_lat(self, magnetic_latitude_deg: float) -> float:
        """Compute the geomagnetic latitude enhancement factor."""

        normalized = min(abs(magnetic_latitude_deg) / 90.0, 1.0)
        return 1.0 + self.a * math.pow(normalized, self.beta)

    def f_alt(self, altitude_km: float) -> float:
        """Compute the altitude factor relative to reference altitude h0."""

        return 1.0 + self.c * ((altitude_km - self.h0) / self.h0)

    def compute_p_seu(self, magnetic_latitude_deg: float, altitude_km: float) -> float:
        """Compute single-step SEU failure probability for a satellite."""

        exponent = -self.k0 * self.S * self.f_lat(magnetic_latitude_deg) * self.f_alt(altitude_km) * self.dt
        prob = 1.0 - math.exp(exponent)
        return min(max(prob, 0.0), 1.0)

    def update(self, sim_state: object, t: int) -> List[str]:
        """Advance the Markov chain for all satellites one time step.

        ``sim_state`` can be the LEONetworkModel instance, a constellation
        container with a ``satellites`` attribute, or a plain iterable of
        satellite-like objects. The model will mutate node/link availability
        inline by marking node status and activity flags when available and by
        disconnecting failed satellites from the underlying LEOCraft graph so
        routing/performance metrics will observe the outage.
        """

        # Respect configured active window so we do not inject outside the threat duration.
        if self.start_time is not None and t < self.start_time:
            return []
        if self.end_time is not None and t > self.end_time:
            return []

        satellites = list(self._extract_satellites(sim_state))
        changed: List[str] = []
        for idx, sat in enumerate(satellites):
            sat_id = self._get_sat_id(sat, idx)
            magnetic_lat = self._get_magnetic_latitude(sat)
            altitude_km = self._get_altitude_km(sat)

            # Initialize state lazily; 0 = healthy.
            state = self.sat_state.get(sat_id, 0)
            p_seu = self.compute_p_seu(magnetic_lat, altitude_km)

            if state == 0:
                # Satellite is healthy; sample potential failure.
                if self.rng.random() < p_seu:
                    self.sat_state[sat_id] = 1
                    self._set_status(sat, "failed")
                    self._mark_graph_node(sim_state, sat_id, active=False)
                    self._disconnect_satellite(sim_state, sat_id)
                    changed.append(sat_id)
                else:
                    self._set_status(sat, "healthy")
                    self._mark_graph_node(sim_state, sat_id, active=True)
            else:
                # Satellite is failed; sample recovery.
                if self.rng.random() < self.q_rec:
                    self.sat_state[sat_id] = 0
                    self._set_status(sat, "healthy")
                    self._mark_graph_node(sim_state, sat_id, active=True)
                    self._restore_satellite(sim_state, sat_id)
                    changed.append(sat_id)
                else:
                    self._set_status(sat, "failed")
                    self._mark_graph_node(sim_state, sat_id, active=False)
                    self._disconnect_satellite(sim_state, sat_id)

        return changed

    # Internal helpers -------------------------------------------------
    def _get_sat_id(self, sat: object, fallback_index: int) -> str:
        sat_id = getattr(sat, "id", None)
        if sat_id is None and isinstance(sat, dict):
            sat_id = sat.get("id")
        return str(sat_id) if sat_id is not None else f"sat_{fallback_index}"

    def _extract_satellites(self, sim_state: object) -> Iterable[object]:
        """Resolve the satellite container from the sim_state."""

        if isinstance(sim_state, dict) and "satellites" in sim_state:
            return sim_state["satellites"]

        for attr in ("satellites", "nodes", "satellite_list"):
            if hasattr(sim_state, attr):
                sats = getattr(sim_state, attr)
                if sats is not None:
                    return sats

        artifacts = getattr(sim_state, "artifacts", None)
        if artifacts is not None:
            constellation = getattr(artifacts, "constellation", None)
            for attr in ("satellites", "satellite_list"):
                sats = getattr(constellation, attr, None)
                if sats is not None:
                    return sats

        graph = self._get_graph(sim_state)
        if graph is not None:
            try:
                return [data if data else node for node, data in graph.nodes(data=True)]
            except Exception:
                return []

        if isinstance(sim_state, Iterable) and not isinstance(sim_state, (str, bytes)):
            return sim_state

        return []

    def _get_magnetic_latitude(self, sat: object) -> float:
        for attr in ("magnetic_latitude", "mag_latitude", "mag_lat", "latitude", "lat"):
            val = getattr(sat, attr, None) if not isinstance(sat, dict) else sat.get(attr)
            if val is not None:
                return float(val)
        return 0.0

    def _get_altitude_km(self, sat: object) -> float:
        for attr in ("altitude_km", "altitude", "height_km", "height"):
            val = getattr(sat, attr, None) if not isinstance(sat, dict) else sat.get(attr)
            if val is not None:
                return float(val)
        return self.h0

    def _set_status(self, sat: object, status: str) -> None:
        if isinstance(sat, dict):
            sat["status"] = status
        else:
            setattr(sat, "status", status)
        # Keep an "active" boolean aligned with status when available.
        try:
            setattr(sat, "active", status != "failed")
        except Exception:
            pass

    def _mark_graph_node(self, sim_state: object, sat_id: str, *, active: bool) -> None:
        """Set node attributes in the underlying graph to mirror outage state."""

        graph = self._get_graph(sim_state)
        if graph is None:
            return
        try:
            if sat_id in graph.nodes:
                graph.nodes[sat_id]["active"] = active
                graph.nodes[sat_id]["status"] = "failed" if not active else "healthy"
        except Exception:
            return

    def _disconnect_satellite(self, sim_state: object, sat_id: str) -> None:
        """Physically remove incident edges so LEOCraft routing loses paths."""

        graph = self._get_graph(sim_state)
        if graph is None:
            return
        if sat_id not in graph.nodes:
            return
        try:
            incident = list(graph.edges(sat_id))
            if incident:
                # Cache edges so they can be restored if the satellite recovers.
                self._disabled_edges.setdefault(sat_id, [])
                # Avoid duplicating edges in the cache across repeated failures.
                existing = set(self._disabled_edges[sat_id])
                for edge in incident:
                    ordered = tuple(edge)
                    if ordered not in existing:
                        self._disabled_edges[sat_id].append(ordered)
                graph.remove_edges_from(incident)
        except Exception:
            return

    def _restore_satellite(self, sim_state: object, sat_id: str) -> None:
        """Reattach previously removed ISLs when a node recovers."""

        graph = self._get_graph(sim_state)
        if graph is None:
            return
        cached = self._disabled_edges.get(sat_id, [])
        if not cached:
            return
        try:
            for edge in cached:
                if edge[0] not in graph.nodes or edge[1] not in graph.nodes:
                    continue
                if not graph.has_edge(*edge):
                    graph.add_edge(*edge)
            self._disabled_edges[sat_id] = []
        except Exception:
            return

    def _get_graph(self, sim_state: object):
        graph = getattr(sim_state, "graph", None)
        if graph is not None:
            return graph
        accessor = getattr(sim_state, "_leocraft_graph", None)
        if callable(accessor):
            try:
                return accessor()
            except Exception:
                return None
        return None


@dataclass
class Satellite:
    """Minimal satellite representation used in the demo simulation loop."""

    id: str
    magnetic_latitude: float
    altitude_km: float
    status: str = "healthy"
    active: bool = True
    neighbors: List[str] = field(default_factory=list)


def _build_link_status(satellites: List[Satellite]) -> Dict[Tuple[str, str], bool]:
    """Create an adjacency map for ISL availability."""

    link_status: Dict[Tuple[str, str], bool] = {}
    for sat in satellites:
        for neighbor in sat.neighbors:
            edge = tuple(sorted((sat.id, neighbor)))
            link_status.setdefault(edge, True)
    return link_status


def _disable_all_links_of_satellite(
    sat_id: str, link_status: Dict[Tuple[str, str], bool]
) -> None:
    for edge in list(link_status.keys()):
        if sat_id in edge:
            link_status[edge] = False


def _enable_all_links_of_satellite(
    sat_id: str, link_status: Dict[Tuple[str, str], bool]
) -> None:
    for edge in list(link_status.keys()):
        if sat_id in edge:
            link_status[edge] = True


def run_demo_simulation(num_steps: int = 10) -> None:
    """Example main loop wiring the solar storm model into a LEO network."""

    # Demo topology with four satellites and a simple ring of ISLs.
    satellites = [
        Satellite("SAT-1", magnetic_latitude=0.0, altitude_km=550.0, neighbors=["SAT-2", "SAT-4"]),
        Satellite("SAT-2", magnetic_latitude=35.0, altitude_km=550.0, neighbors=["SAT-1", "SAT-3"]),
        Satellite("SAT-3", magnetic_latitude=60.0, altitude_km=550.0, neighbors=["SAT-2", "SAT-4"]),
        Satellite("SAT-4", magnetic_latitude=-20.0, altitude_km=550.0, neighbors=["SAT-1", "SAT-3"]),
    ]

    link_status = _build_link_status(satellites)
    sat_by_id = {sat.id: sat for sat in satellites}

    # === Solar storm induced node outage (Monte Carlo, physically-based) ===
    storm_model = SolarStormNodeOutageModel(
        S=500,  # Storm intensity; can be externalized to config
        k0=3e-7,
        a=4.0,
        beta=1.5,
        c=0.2,
        h0=550.0,
        dt=1.0,
        q_rec=0.05,  # Per-step recovery probability
    )

    for t in range(num_steps):
        print(f"\n[SIM] Timestep {t}")

        # Update satellite health based on current geomagnetic latitude/altitude.
        changed_sat_ids = storm_model.update(satellites, t)

        # 根据 storm_model 的内部状态，更新卫星与链路的可用性。
        for sat_id in changed_sat_ids:
            sat = sat_by_id[sat_id]
            if storm_model.sat_state[sat_id] == 1:
                # Satellite failure: deactivate node and close all connected ISLs.
                sat.active = False
                _disable_all_links_of_satellite(sat_id, link_status)
                print(f"[WARN] {sat_id} failed due to solar storm; disabling adjacent ISLs")
            else:
                # Satellite recovery: re-enable node and reopen ISLs.
                sat.active = True
                _enable_all_links_of_satellite(sat_id, link_status)
                print(f"[INFO] {sat_id} recovered; re-enabling adjacent ISLs")

        # Placeholder for routing/metrics update using current link_status.
        active_nodes = [sat.id for sat in satellites if sat.active]
        failed_nodes = [sat.id for sat in satellites if not sat.active]
        active_links = [edge for edge, enabled in link_status.items() if enabled]
    print(
        "[STATE] Active nodes:", active_nodes,
        "| Failed nodes:", failed_nodes,
        "| Active ISLs:", active_links,
    )
