from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage


@dataclass(slots=True)
class Session:
    """Historial aislado mientras un turno posee la sesión."""

    session_id: str
    messages: list[ModelMessage]
    _changed: bool = field(default=False, init=False, repr=False)

    @property
    def changed(self) -> bool:
        return self._changed

    def replace(self, messages: Sequence[ModelMessage]) -> None:
        self.messages = list(messages)
        self._changed = True
