"""Optional Langfuse tracing and score reporting."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any


def langfuse_is_configured() -> bool:
    """Return whether enough environment is present to send Langfuse data."""

    import os

    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_BASE_URL")
    )


def create_langfuse_handler() -> Any | None:
    """Create the LangChain callback handler, or disable tracing cleanly."""

    if not langfuse_is_configured():
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


@contextmanager
def trace_agent_run(
    *,
    handler: Any | None,
    trace_name: str,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[tuple[Any | None, str | None]]:
    """Create a Langfuse root span and yield it with its trace ID."""

    if handler is None:
        with nullcontext():
            yield None, None
        return

    from langfuse import get_client, propagate_attributes

    langfuse = get_client()
    with propagate_attributes(
        trace_name=trace_name,
        tags=list(tags),
        metadata=dict(metadata or {}),
    ):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="agent-run",
        ) as root_span:
            yield root_span, root_span.trace_id


def flush_langfuse() -> None:
    """Flush queued Langfuse events without making tracing mandatory."""

    if not langfuse_is_configured():
        return

    from langfuse import get_client

    get_client().flush()


def record_score(
    *,
    trace_id: str | None,
    name: str,
    value: float | int | bool,
    comment: str | None = None,
) -> None:
    """Attach a DeepEval score to an existing Langfuse trace."""

    if trace_id is None or not langfuse_is_configured():
        return

    from langfuse import get_client

    get_client().create_score(
        name=name,
        value=float(value),
        trace_id=trace_id,
        data_type="NUMERIC",
        comment=comment,
    )
    flush_langfuse()
