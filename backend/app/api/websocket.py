"""
app/api/websocket.py
────────────────────
Real-time WebSocket endpoint streaming structured inference events to clients.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.events import DataSource, EventType, InferenceEvent
from app.core.session import session_manager
from app.inference.engine import inference_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{session_id}")
async def websocket_inference_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket connection handler for a specific inference session."""
    await websocket.accept()

    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_json({
            "error": "Session not found",
            "session_id": session_id,
        })
        await websocket.close(code=4004)
        return

    logger.info("WebSocket connected for session: %s", session_id)

    async def send_event(event: InferenceEvent) -> None:
        """Deliver event to frontend client over WebSocket."""
        try:
            # Model dump json handles all serializations cleanly
            await websocket.send_text(event.model_dump_json())
        except Exception as exc:
            logger.warning("Failed to send WebSocket event: %s", exc)

    try:
        # Run inference through the engine
        await inference_engine.run(
            session_id=session_id,
            model_id=session.model_id,
            prompt=session.prompt,
            config=session.config,
            ws_callback=send_event,
        )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for session: %s", session_id)
        if session.status == "running":
            session.status = "paused"
    except Exception as exc:
        logger.exception("Unexpected error in websocket loop: %s", exc)
        try:
            err_event = InferenceEvent(
                session_id=session_id,
                event_id="err-unhandled",
                event_type=EventType.INFERENCE_ERROR,
                sequence_number=len(session.events) + 1,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                stage="error",
                data={"message": str(exc), "error_type": type(exc).__name__},
                data_source=DataSource.ACTUAL,
            )
            await websocket.send_text(err_event.model_dump_json())
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

