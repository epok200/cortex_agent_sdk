from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cortex_agent_sdk.engine import ToolCall, Usage
from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.history.models import Turn
from cortex_agent_sdk.immutable import JsonObject
from cortex_agent_sdk.tools.execution import ToolOutcome


class ApprovalAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ApprovalAction
    rejection_message: str | None = None


class ToolApproval(BaseModel):
    """Llamada sensible que espera una decisión humana antes de ejecutarse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    tool_name: str
    arguments: JsonObject

    @classmethod
    def from_call(cls, call: ToolCall) -> "ToolApproval":
        return cls(call_id=call.call_id, tool_name=call.name, arguments=call.arguments)


class PendingRun(BaseModel):
    """Snapshot serializable suficiente para reanudar un run pausado por aprobación."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    provider: str
    model: str
    instructions: str
    history: tuple[Turn, ...]
    step: int = Field(ge=1)
    usage: Usage = Field(default_factory=Usage)
    tool_calls: int = Field(default=0, ge=0)
    consecutive_tool_failures: int = Field(default=0, ge=0)
    last_text: str | None = None
    fallback_text: str | None = None
    effective_model: str | None = None
    session_id: str | None = None
    calls: tuple[ToolCall, ...]
    interruptions: tuple[ToolApproval, ...]
    decisions: dict[str, ApprovalDecision] = Field(default_factory=dict)
    response_model_name: str | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        call_ids = tuple(call.call_id for call in self.calls)
        interruption_ids = {item.call_id for item in self.interruptions}
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("calls contiene call_id duplicado")
        if not interruption_ids or not interruption_ids <= set(call_ids):
            raise ValueError("interruptions no coincide con calls")
        if not set(self.decisions) <= interruption_ids:
            raise ValueError("decisions contiene call_id sin aprobación pendiente")
        return self

    @property
    def unresolved(self) -> tuple[ToolApproval, ...]:
        return tuple(item for item in self.interruptions if item.call_id not in self.decisions)

    def approve(self, call_id: str) -> None:
        self._require_interruption(call_id)
        self.decisions[call_id] = ApprovalDecision(action=ApprovalAction.APPROVE)

    def reject(self, call_id: str, *, message: str | None = None) -> None:
        self._require_interruption(call_id)
        self.decisions[call_id] = ApprovalDecision(
            action=ApprovalAction.REJECT,
            rejection_message=message,
        )

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, value: str) -> "PendingRun":
        return cls.model_validate_json(value)

    def _require_interruption(self, call_id: str) -> None:
        if not any(item.call_id == call_id for item in self.interruptions):
            raise AppError(CodigoError.CONFIG_INVALIDA, f"call no requiere aprobación: {call_id}")


@dataclass(frozen=True, slots=True)
class ContinueRun:
    """Decisión de policy que devuelve los resultados al modelo y continúa el loop."""


CONTINUE = ContinueRun()


@dataclass(frozen=True, slots=True)
class FinalOutput:
    """Decisión de policy que termina el run con un output explícito."""

    output: str

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise TypeError("FinalOutput.output debe ser str")


@dataclass(frozen=True, slots=True)
class ToolResultContext:
    """Contexto mínimo disponible para una policy post-tool."""

    step: int
    session_id: str | None
    history: tuple[Turn, ...]
    outcomes: tuple[ToolOutcome, ...]


type ToolResultDecision = ContinueRun | FinalOutput
type ToolResultPolicy = Callable[
    [ToolResultContext],
    ToolResultDecision | Awaitable[ToolResultDecision],
]
