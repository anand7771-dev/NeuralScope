"""
app/inference/engine.py
───────────────────────
Central coordinator for model registry and inference execution.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.adapters.base import AbstractModelAdapter, ModelInfo
from app.core.events import DataSource, EventType, InferenceEvent
from app.core.session import session_manager

__all__ = ["InferenceEngine", "inference_engine"]

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages model adapters and orchestrates inference sessions."""

    def __init__(self) -> None:
        self._adapters: Dict[str, AbstractModelAdapter] = {}

    def register_adapter(self, model_id: str, adapter: AbstractModelAdapter) -> None:
        """Register a model adapter under a unique identifier."""
        self._adapters[model_id] = adapter
        logger.info("Registered adapter for model: %s", model_id)

    def get_adapter(self, model_id: str) -> Optional[AbstractModelAdapter]:
        """Retrieve an adapter by model ID."""
        return self._adapters.get(model_id)

    def list_available_models(self) -> List[ModelInfo]:
        """Return model metadata for all registered adapters."""
        return [adapter.get_info() for adapter in self._adapters.values()]

    async def run(
        self,
        session_id: str,
        model_id: str,
        prompt: str,
        config: Dict[str, Any],
        ws_callback: Callable[[InferenceEvent], Awaitable[None]],
    ) -> Dict[str, Any]:
        """Execute instrumented inference for a session."""
        session = session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found.")

        adapter = self.get_adapter(model_id)
        if not adapter:
            err_event = InferenceEvent(
                session_id=session_id,
                event_id="err-missing-model",
                event_type=EventType.INFERENCE_ERROR,
                sequence_number=9999,
                timestamp="",
                stage="error",
                data={"message": f"Model adapter '{model_id}' is not registered."},
                data_source=DataSource.ACTUAL,
            )
            await ws_callback(err_event)
            session.fail(f"Model adapter '{model_id}' not registered")
            return {"error": f"Model adapter '{model_id}' not registered"}

        session.start()

        # Intercept events to keep session state synchronized
        async def session_sync_callback(event: InferenceEvent) -> None:
            if event.event_type == EventType.TOKEN_GENERATED:
                token_str = event.data.get("token_string", "")
                session.append_token(token_str)
            elif event.event_type == EventType.GENERATION_COMPLETED:
                session.complete()
            elif event.event_type == EventType.INFERENCE_ERROR:
                session.fail(event.data.get("message", "Inference error"))

            await ws_callback(event)

        try:
            # Run the adapter's instrumented inference loop
            result = await adapter.run_instrumented_inference(
                prompt=prompt,
                config=config,
                event_callback=session_sync_callback,
                session_id=session_id,
            )
            return result
        except Exception as exc:
            logger.exception("Inference failed for session %s: %s", session_id, exc)
            session.fail(str(exc))
            err_event = InferenceEvent(
                session_id=session_id,
                event_id="err-inference-failure",
                event_type=EventType.INFERENCE_ERROR,
                sequence_number=len(session.events) + 1,
                timestamp="",
                stage="error",
                data={"message": str(exc), "error_type": type(exc).__name__},
                data_source=DataSource.ACTUAL,
            )
            await ws_callback(err_event)
            return {"error": str(exc)}


inference_engine = InferenceEngine()

