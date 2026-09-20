import json
from typing import Any

from backend.app.core.logging import logger


def format_sse_event(event_name: str, data: Any) -> str:
    """Formats data as a valid Server-Sent Events (SSE) packet."""
    try:
        if isinstance(data, (dict, list)):
            payload = json.dumps(data, default=str)
        else:
            payload = str(data)
        return f"event: {event_name}\ndata: {payload}\n\n"
    except Exception as e:
        logger.error(f"Error formatting SSE event '{event_name}': {e}", exc_info=True)
        return f"event: {event_name}\ndata: {str(data)}\n\n"


def format_done_event() -> str:
    """Returns the terminal SSE termination event."""
    try:
        return "event: end\ndata: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error formatting SSE done event: {e}", exc_info=True)
        return "event: end\ndata: [DONE]\n\n"
