"""Solar storm induced satellite outage model.

This module provides a Monte Carlo based disturbance model that simulates
single event upsets (SEU) triggered by solar storms. It follows the provided
physics-inspired probability functions and exposes a deterministic update
interface that can be invoked from the LEO network simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class SolarStormNodeOutageModel:
    S: float
    q_rec: float
    k0: float = 3e-7
    a: float = 4.0
    beta: float = 1.5
    c: float = 0.2
    h0: float = 550.0
    dt: float = 1.0
    rng: random.Random = field(default_factory=random.Random)
    sat_state: Dict[str, int] = field(default_factory=dict)

    def f_lat(self, magnetic_latitude_deg: float) -> float:
        """Compute the geomagnetic latitude enhancement factor.

        f_lat(lambda_m) = 1 + a * (|lambda_m| / 90)^beta
        """

        normalized = min(abs(magnetic_latitude_deg) / 90.0, 1.0)
        return 1.0 + self.a * math.pow(normalized, self.beta)

    def f_alt(self, altitude_km: float) -> float:
        """Compute the altitude factor relative to reference altitude h0.

        f_alt(h) = 1 + c * (h - h0) / h0
        """

        return 1.0 + self.c * ((altitude_km - self.h0) / self.h0)

    def compute_p_seu(self, magnetic_latitude_deg: float, altitude_km: float) -> float:
        """Compute single-step SEU failure probability for a satellite.

        p_SEU = 1 - exp(-k0 * S * f_lat(lambda_m) * f_alt(h) * dt)
        The value is clamped into [0, 1] to guard against numerical drift.
        """

        exponent = -self.k0 * self.S * self.f_lat(magnetic_latitude_deg) * self.f_alt(altitude_km) * self.dt
        prob = 1.0 - math.exp(exponent)
        return min(max(prob, 0.0), 1.0)

    def update(self, satellites: Iterable[object], t: int) -> List[str]:
        """Advance the Markov chain for all satellites one time step.

        Args:
            satellites: Iterable of satellite-like objects exposing ``id``,
                ``magnetic_latitude`` (or ``lat``/``latitude`` fallback) and
                ``altitude_km`` (or ``altitude``) attributes. Each satellite
                may also carry a ``status`` field that will be updated to
                ``"failed"`` or ``"healthy"``.
            t: Current simulation timestep (kept for interface compatibility).

        Returns:
            List of satellite IDs whose status flipped during this update.
        """

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
                    changed.append(sat_id)
                else:
                    self._set_status(sat, "healthy")
            else:
                # Satellite is failed; sample recovery.
                if self.rng.random() < self.q_rec:
                    self.sat_state[sat_id] = 0
                    self._set_status(sat, "healthy")
                    changed.append(sat_id)
                else:
                    self._set_status(sat, "failed")

        return changed

    # Internal helpers -------------------------------------------------
    def _get_sat_id(self, sat: object, fallback_index: int) -> str:
        sat_id = getattr(sat, "id", None)
        if sat_id is None and isinstance(sat, dict):
            sat_id = sat.get("id")
        return str(sat_id) if sat_id is not None else f"sat_{fallback_index}"

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