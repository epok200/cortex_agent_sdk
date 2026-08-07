from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from cortex_agent_sdk.history.models import Turn


@dataclass(frozen=True, slots=True)
class ToolApprovalContext:
    """Contexto mínimo para decidir si una tool call necesita aprobación."""

    step: int
    session_id: str | None
    history: tuple[Turn, ...]


type ToolApprovalPredicate = Callable[
    [ToolApprovalContext, Mapping[str, object], str],
    bool | Awaitable[bool],
]
type ToolApprovalRule = bool | ToolApprovalPredicate
