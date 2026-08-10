from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from cortex_agent_sdk.errores import AppError, CodigoError


def encode_messages(messages: Sequence[ModelMessage]) -> bytes:
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def decode_messages(payload: bytes) -> list[ModelMessage]:
    try:
        return ModelMessagesTypeAdapter.validate_json(payload)
    except ValueError as error:
        raise AppError(CodigoError.SESION_INVALIDA, "historial de sesión inválido") from error
