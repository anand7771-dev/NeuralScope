"""
app/core/session.py
───────────────────
Session management for NeuralScope inference runs.

Each WebSocket connection is backed by an ``InferenceSession`` that holds the
ordered list of emitted events (for replay), the generated text, and all
timing / configuration metadata.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.core.events import InferenceEvent

__all__ = [
    "InferenceSession",
    "SessionManager",
    "session_manager",
]

logger = logging.getLogger(__name__)


# ── Session model ─────────────────────────────────────────────────────────────


class InferenceSession:
    """Represents a single inference run from prompt to completion.

    Instances are created by ``SessionManager`` and are held in memory for
    their lifetime. Event replay is possible because every emitted
    ``InferenceEvent`` is appended to ``self.events``.
    """

    def __init__(
        self,
        *,
        session_id: str,
        model_id: str,
        prompt: str,
        config: Dict[str, Any],
    ) -> None:
        self.session_id: str = session_id
        self.model_id: str = model_id
        self.prompt: str = prompt
        self.config: Dict[str, Any] = config

        # Mutable state
        self.status: str = "idle"  # idle | running | paused | completed | error
        self.events: List[InferenceEvent] = []
        self.generated_text: str = ""
        self.token_count: int = 0
        self.total_duration_ms: float = 0.0

        # Timestamps
        self.created_at: datetime = datetime.now(tz=timezone.utc)
        self.completed_at: Optional[datetime] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Alias for mark_running."""
        self.mark_running()

    def mark_running(self) -> None:
        """Transition to the *running* state."""
        self.status = "running"

    def append_token(self, token_str: str) -> None:
        """Record an incrementally generated token."""
        self.generated_text += token_str
        self.token_count += 1

    def complete(self) -> None:
        """Mark session completed."""
        self.status = "completed"
        self.completed_at = datetime.now(tz=timezone.utc)

    def fail(self, reason: str = "") -> None:
        """Mark session as failed."""
        self.mark_error()

    def mark_completed(
        self,
        generated_text: str,
        token_count: int,
        total_duration_ms: float,
    ) -> None:
        """Transition to the *completed* state and record final metrics."""
        self.status = "completed"
        self.generated_text = generated_text
        self.token_count = token_count
        self.total_duration_ms = total_duration_ms
        self.completed_at = datetime.now(tz=timezone.utc)

    def mark_error(self) -> None:
        """Transition to the *error* state."""
        self.status = "error"
        self.completed_at = datetime.now(tz=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary (without the full events list)."""
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "status": self.status,
            "prompt": self.prompt,
            "config": self.config,
            "generated_text": self.generated_text,
            "token_count": self.token_count,
            "total_duration_ms": self.total_duration_ms,
            "event_count": len(self.events),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ── Session manager ───────────────────────────────────────────────────────────


class SessionManager:
    """Manages lifecycle of all in-memory ``InferenceSession`` instances."""

    def __init__(self, max_sessions: int = 10) -> None:
        self._sessions: Dict[str, InferenceSession] = {}
        self._max_sessions: int = max_sessions

    def create_session(
        self,
        *,
        model_id: str,
        prompt: str,
        config: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> InferenceSession:
        """Create a new session, evicting the oldest if limit reached."""
        if len(self._sessions) >= self._max_sessions:
            oldest_id = next(iter(self._sessions))
            self.delete_session(oldest_id)

        sid = session_id or str(uuid.uuid4())
        session = InferenceSession(
            session_id=sid,
            model_id=model_id,
            prompt=prompt,
            config=config,
        )
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[InferenceSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[InferenceSession]:
        return list(self._sessions.values())

    def delete_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session:
            try:
                from app.core.tensor_store import tensor_store
                tensor_store.clear_session(session_id)
            except Exception as exc:
                logger.warning("Error clearing tensors for session %s: %s", session_id, exc)
            return True
        return False


_settings = get_settings()
session_manager = SessionManager(max_sessions=_settings.max_sessions)
