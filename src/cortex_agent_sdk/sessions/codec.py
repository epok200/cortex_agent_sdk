from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.sessions.models import SessionRecord


class _SessionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    record: SessionRecord


class SessionCodec:
    """Serializa sesiones a JSON validado, sin pickle."""

    def encode(self, record: SessionRecord) -> bytes:
        return _SessionEnvelope(record=record).model_dump_json().encode()

    def decode(self, payload: bytes) -> SessionRecord:
        try:
            return _SessionEnvelope.model_validate_json(payload).record
        except ValidationError as error:
            raise AppError(CodigoError.SESION_INVALIDA, "payload de sesión inválido") from error

