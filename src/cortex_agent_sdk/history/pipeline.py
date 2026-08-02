from datetime import UTC, datetime

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.history.models import Turn
from cortex_agent_sdk.history.transform import HistoryTransform, TransformContext


class HistoryPipeline:
    def __init__(self, transform: HistoryTransform | None, max_turns: int) -> None:
        self._transform = transform
        self._max_turns = max_turns

    async def prepare(
        self,
        history: tuple[Turn, ...],
        session_id: str | None,
    ) -> tuple[Turn, ...]:
        transformed = history
        if self._transform is not None:
            context = TransformContext(
                history=history,
                max_turns=self._max_turns,
                now=datetime.now(UTC),
                session_id=session_id,
            )
            try:
                transformed = tuple(await self._transform(context))
            except AppError:
                raise
            except Exception as error:
                raise AppError(
                    CodigoError.HISTORIAL_TRANSFORM_INVALIDO,
                    f"history_transform lanzó {type(error).__name__}",
                ) from error
            if not all(isinstance(turn, Turn) for turn in transformed):
                raise AppError(
                    CodigoError.HISTORIAL_TRANSFORM_INVALIDO,
                    "history_transform devolvió un tipo inválido",
                )
        return _window_history(transformed, self._max_turns)

    def window(self, history: tuple[Turn, ...]) -> tuple[Turn, ...]:
        return _window_history(history, self._max_turns)


def _window_history(history: tuple[Turn, ...], max_turns: int) -> tuple[Turn, ...]:
    if len(history) <= max_turns:
        return history
    start = len(history) - max_turns
    while history[start].role == "tool" and start > 0:
        start -= 1
    return history[start:]

