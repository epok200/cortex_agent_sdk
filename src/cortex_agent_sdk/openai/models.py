from collections.abc import Callable
from typing import Literal

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from cortex_agent_sdk.errores import AppError, CodigoError


class OpenAIModel(BaseModel):
    """Modelo disponible según el catálogo nativo de OpenAI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    created: int
    object: Literal["model"]
    owned_by: str


_OPENAI_MODELS = TypeAdapter(tuple[OpenAIModel, ...])


class OpenAIModels:
    """Consulta el catálogo con el mismo cliente que usa el engine."""

    def __init__(self, client: AsyncOpenAI, ensure_open: Callable[[], None]) -> None:
        self._client = client
        self._ensure_open = ensure_open

    async def list(self) -> tuple[OpenAIModel, ...]:
        self._ensure_open()
        try:
            response = await self._client.models.list()
        except openai.APITimeoutError as error:
            raise AppError(CodigoError.PROVIDER_TIMEOUT, "OpenAI agotó el timeout") from error
        except openai.APIStatusError as error:
            request_id = getattr(error, "request_id", None)
            detail = f"OpenAI respondió {error.status_code}, request_id={request_id}"
            raise AppError(CodigoError.PROVIDER_FALLO, detail) from error
        except openai.APIConnectionError as error:
            raise AppError(CodigoError.PROVIDER_FALLO, "falló la conexión con OpenAI") from error
        except openai.APIError as error:
            detail = f"OpenAI SDK lanzó {type(error).__name__}"
            raise AppError(CodigoError.PROVIDER_FALLO, detail) from error

        try:
            return _OPENAI_MODELS.validate_python(
                response.data,
                from_attributes=True,
            )
        except ValidationError as error:
            raise AppError(
                CodigoError.PROVIDER_RESPUESTA_INVALIDA,
                "OpenAI devolvió un catálogo de modelos inválido",
            ) from error
