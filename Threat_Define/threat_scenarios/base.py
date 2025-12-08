"""Base interfaces for threat scenario definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Protocol


@dataclass
class ScenarioContext:
    """Contextual information describing the LEO network state."""

    satellite_count: int
    inter_satellite_links: int
    ground_stations: int
    critical_services: List[str]


class ThreatScenario(ABC):
    """Abstract base class representing a risk scenario for the LEO network."""

    name: str
    category: str

    def __init__(self, name: str, category: str | None = None, **_: object) -> None:
        """
        Initialize the threat scenario.

        Some callers may still pass legacy positional arguments, so we accept an
        optional ``category`` as well as stray keyword arguments to stay
        backward-compatible with older agent wiring. If no category is
        provided, we fall back to ``"uncategorized"`` to keep downstream
        serialization stable.
        """

        self.name = name
        self.category = category or "uncategorized"

    @abstractmethod
    def generate(self, context: ScenarioContext) -> Dict[str, object]:
        """Produce a concrete threat description using the provided context."""

    @abstractmethod
    def apply(self, network: "LEONetwork", payload: Dict[str, object]) -> None:
        """Inject the threat payload into the provided LEO network simulation."""

    @abstractmethod
    def key_parameters(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Extract the essential parameters that define this threat instance."""


class LEONetwork(Protocol):
    """Protocol describing minimal operations required by the scenarios."""

    def inject_disturbance(self, description: str, impact: Dict[str, object]) -> None:
        """Apply a disturbance into the network model."""

    def log_event(self, event: str) -> None:
        """Persist a log entry into the network's timeline."""