from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from cortex_agent_sdk.history.models import Turn


@dataclass(frozen=True, slots=True)
class TransformContext:
    history: tuple[Turn, ...]
    max_turns: int
    now: datetime
    session_id: str | None


HistoryTransform = Callable[[TransformContext], Awaitable[Sequence[Turn]]]
