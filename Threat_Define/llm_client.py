"""Utilities for invoking an LLM to create dynamic threat scenarios."""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, TYPE_CHECKING
import importlib.util

_dashscope_spec = importlib.util.find_spec("dashscope")
if _dashscope_spec is not None:  # pragma: no cover - optional dependency branch
    import dashscope  # type: ignore[import]
    from dashscope import Generation  # type: ignore[import]
else:  # pragma: no cover - optional dependency branch
    dashscope = None  # type: ignore[assignment]
    Generation = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from threat_scenarios.base import ScenarioContext


class LLMScenarioGenerator:
    """Helper that wraps calls to an LLM for scenario synthesis."""

    def __init__(
        self,
        model: str = "qwen-turbo",
        temperature: float = 0.2,
        *,
        enable_fallback_metadata: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.enable_fallback_metadata = enable_fallback_metadata
        self._llm_available = False

        if Generation is not None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if api_key:
                dashscope.api_key = api_key
                self._llm_available = True

    def generate_threat(
        self,
        *,
        topic: str,
        context: "ScenarioContext",
        guidance: str,
        schema: Dict[str, Any],
        fallback: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a structured payload for the given threat topic."""

        prompt = self._build_prompt(topic, context, guidance, schema)
        if not self._llm_available:
            payload = fallback()
            return self._augment(payload, used_llm=False)

        try:
            response = Generation.call(
                model=self.model,
                input={
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a space systems red-team assistant that designs LEO "
                                "network attack scenarios. Reply strictly with JSON that matches the "
                                "provided schema."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                },
                temperature=self.temperature,
            )

            status_code = getattr(response, "status_code", 200)
            if status_code and status_code != 200:
                error_message = getattr(response, "message", "LLM request failed")
                raise RuntimeError(f"LLM request failed with status {status_code}: {error_message}")

            raw_text = self._extract_text(response)
            payload = json.loads(raw_text)
            return self._augment(payload, used_llm=True)
        except Exception as exc:  # pragma: no cover - defensive path
            payload = fallback()
            if self.enable_fallback_metadata:
                payload["llm_error"] = str(exc)
            return self._augment(payload, used_llm=False)

    def _build_prompt(
        self,
        topic: str,
        context: "ScenarioContext",
        guidance: str,
        schema: Dict[str, Any],
    ) -> str:
        services = ", ".join(context.critical_services)
        schema_json = json.dumps(schema)
        prompt = (
            f"Design a high-impact {topic} threat scenario for a LEO satellite network.\n"
            f"Network details: {context.satellite_count} satellites, "
            f"{context.inter_satellite_links} inter-satellite links, "
            f"{context.ground_stations} ground stations. Critical services: {services}.\n"
            f"Guidance: {guidance}.\n"
            "Respond strictly with a JSON object that matches this JSON schema: "
            f"{schema_json}."
        )
        return prompt

    def _extract_text(self, response: Any) -> str:
        """Extract the assistant text from a DashScope response."""

        def search(payload: Any) -> str | None:
            if isinstance(payload, str):
                return payload
            if isinstance(payload, dict):
                if isinstance(payload.get("text"), str):
                    return payload["text"]
                for key in ("content", "message", "messages", "choices", "output"):
                    if key in payload:
                        found = search(payload[key])
                        if found:
                            return found
                for value in payload.values():
                    found = search(value)
                    if found:
                        return found
            elif isinstance(payload, list):
                for item in payload:
                    found = search(item)
                    if found:
                        return found
            return None

        candidates = []
        for attribute in ("output", "choices", "message", "content"):
            if hasattr(response, attribute):
                candidates.append(getattr(response, attribute))

        if isinstance(response, dict):
            candidates.append(response)

        for candidate in candidates:
            found_text = search(candidate)
            if found_text:
                return found_text

        raise ValueError("LLM response did not include a text payload")

    def _augment(self, payload: Dict[str, Any], *, used_llm: bool) -> Dict[str, Any]:
        if self.enable_fallback_metadata:
            payload.setdefault("generated_by", "llm" if used_llm else "deterministic")
        return payload


__all__ = ["LLMScenarioGenerator"]