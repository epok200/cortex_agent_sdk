from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import ModelMessage, ToolCallPart

from cortex_agent_sdk.errores import AppError, CodigoError


class ActiveTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: tuple[tuple[str, str], ...]


type _StartToolTurn = Callable[[ActiveTurn], Awaitable[None]]
type _Checkpoint = Callable[[Sequence[ModelMessage]], Awaitable[None]]


@dataclass(slots=True)
class Session:
    """Historial aislado mientras un turno posee la sesión."""

    session_id: str
    messages: list[ModelMessage]
    _changed: bool = field(default=False, init=False, repr=False)
    _start_tool_turn: _StartToolTurn | None = field(default=None, repr=False)
    _save_checkpoint: _Checkpoint | None = field(default=None, repr=False)

    @property
    def changed(self) -> bool:
        return self._changed

    def replace(self, messages: Sequence[ModelMessage]) -> None:
        self.messages = list(messages)
        self._changed = True

    async def start_tool_turn(self, calls: Sequence[ToolCallPart]) -> None:
        if self._start_tool_turn is None:
            raise AppError(CodigoError.CONFIG_INVALIDA, "sesión sin persistencia inmediata")
        if not calls:
            raise AppError(CodigoError.SESION_INVALIDA, "turno de tools activo inválido")

        identities = tuple((call.tool_name, call.tool_call_id) for call in calls)
        active_turn = ActiveTurn(calls=identities)
        await self._start_tool_turn(active_turn)

    async def checkpoint(self, messages: Sequence[ModelMessage]) -> None:
        if self._save_checkpoint is None:
            raise AppError(CodigoError.CONFIG_INVALIDA, "sesión sin persistencia inmediata")

        checkpoint = list(messages)
        await self._save_checkpoint(checkpoint)
        self.messages = checkpoint

    def _detach(self) -> None:
        self._start_tool_turn = None
        self._save_checkpoint = None
