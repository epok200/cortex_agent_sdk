from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.history.models import Turn


class ActiveTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: str
    pending_call_ids: tuple[str, ...]
    completed_call_ids: tuple[str, ...] = ()

    def complete(self, call_id: str) -> "ActiveTurn":
        if call_id not in self.pending_call_ids or call_id in self.completed_call_ids:
            raise AppError(CodigoError.SESION_INVALIDA, f"call inesperada: {call_id}")
        completed = (*self.completed_call_ids, call_id)
        return self.model_copy(update={"completed_call_ids": completed})


class SessionRecord(BaseModel):
    """Estado conversacional versionado y ligado a un engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    provider: str
    model: str
    history: tuple[Turn, ...] = ()
    active_turn: ActiveTurn | None = None
    version: int = 0
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, session_id: str, provider: str, model: str) -> "SessionRecord":
        now = datetime.now(UTC)
        return cls(
            session_id=session_id,
            provider=provider,
            model=model,
            created_at=now,
            updated_at=now,
        )

    def with_history(self, history: tuple[Turn, ...]) -> "SessionRecord":
        return self.model_copy(update={"history": history})

    def start_turn(
        self,
        turn_id: str,
        call_ids: tuple[str, ...],
        history: tuple[Turn, ...],
    ) -> "SessionRecord":
        if self.active_turn is not None or not call_ids or len(call_ids) != len(set(call_ids)):
            raise AppError(CodigoError.SESION_INVALIDA, "turno activo inválido")
        active_turn = ActiveTurn(turn_id=turn_id, pending_call_ids=call_ids)
        return self.model_copy(update={"history": history, "active_turn": active_turn})

    def complete_call(self, call_id: str, history: tuple[Turn, ...]) -> "SessionRecord":
        if self.active_turn is None:
            return self.with_history(history)
        active_turn = self.active_turn.complete(call_id)
        return self.model_copy(update={"history": history, "active_turn": active_turn})

    def finish_turn(self, history: tuple[Turn, ...]) -> "SessionRecord":
        return self.model_copy(update={"history": history, "active_turn": None})

