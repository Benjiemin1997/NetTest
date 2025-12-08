"""Agent interfaces for building multi-agent risk orchestration."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Protocol

from Threat_Define.threat_scenarios import ThreatScenario, ScenarioContext


class ScenarioEvaluator(Protocol):
    """Evaluators score threat payloads to determine priority."""

    def score(self, scenario: ThreatScenario, payload: Dict[str, object]) -> float:
        ...


class RiskAgent(ABC):
    """Base class for autonomous agents collaborating on scenario generation."""

    def __init__(self, name: str, scenario: ThreatScenario, evaluator: ScenarioEvaluator) -> None:
        self.name = name
        self.scenario = scenario
        self.evaluator = evaluator

    @abstractmethod
    def perceive(self, context: ScenarioContext) -> Dict[str, object]:
        """Collect observations and propose a threat payload."""

    def evaluate(self, payload: Dict[str, object]) -> float:
        """Score the payload using the shared evaluator."""
        return self.evaluator.score(self.scenario, payload)

    def act(self, network: "LEONetwork", payload: Dict[str, object]) -> None:
        """Apply the scenario to the provided network."""
        self.scenario.apply(network, payload)


class LEONetwork(Protocol):
    """Protocol for type checking the network passed to agents."""

    def inject_disturbance(self, description: str, impact: Dict[str, object]) -> None:
        ...

    def log_event(self, event: str) -> None:
        ...