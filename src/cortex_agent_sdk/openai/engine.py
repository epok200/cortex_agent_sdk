import asyncio
import json
from typing import Any, Self, cast

import openai
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall, ResponseInputItemParam
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from cortex_agent_sdk.engine import EngineRequest, EngineResult, ToolCall, Usage
from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.gateway import OpenAICompatibleGateway
from cortex_agent_sdk.history.models import (
    ProviderState,
    ProviderValue,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Turn,
)
from cortex_agent_sdk.immutable import (
    FrozenJsonValue,
    FrozenProviderValue,
    thaw_json,
    thaw_provider,
)
from cortex_agent_sdk.openai.options import OpenAIOptions
from cortex_agent_sdk.tools.models import ToolSpec

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_PROVIDER_OBJECT = TypeAdapter(dict[str, ProviderValue])


class OpenAIEngine:
    """Engine de una inferencia por llamada sobre OpenAI Responses."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        gateway: OpenAICompatibleGateway | None = None,
        options: OpenAIOptions | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not model:
            raise AppError(CodigoError.CONFIG_INVALIDA, "model no puede estar vacío")
        if client is not None and (api_key is not None or gateway is not None):
            raise AppError(CodigoError.CONFIG_INVALIDA, "client excluye api_key y gateway")
        if gateway is not None and api_key is not None:
            raise AppError(CodigoError.CONFIG_INVALIDA, "gateway excluye api_key")

        self._model = model
        self._options = options or OpenAIOptions()
        self._client = client or self._create_client(api_key, gateway)
        self._owns_client = client is None
        self._closed = False

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def __aenter__(self) -> Self:
        if self._closed:
            raise AppError(CodigoError.RECURSO_CERRADO, "OpenAIEngine cerrado")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def generate(self, request: EngineRequest) -> EngineResult:
        if self._closed:
            raise AppError(CodigoError.RECURSO_CERRADO, "OpenAIEngine cerrado")

        payload = cast(dict[str, Any], self._build_payload(request))
        try:
            response = await self._client.responses.create(**payload)
        except openai.APITimeoutError as error:
            raise AppError(CodigoError.PROVIDER_TIMEOUT, "OpenAI agotó el timeout") from error
        except openai.APIStatusError as error:
            request_id = getattr(error, "request_id", None)
            detail = f"OpenAI respondió {error.status_code}, request_id={request_id}"
            raise AppError(CodigoError.PROVIDER_FALLO, detail) from error
        except openai.APIConnectionError as error:
            raise AppError(CodigoError.PROVIDER_FALLO, "falló la conexión con OpenAI") from error
        except openai.APIError as error:
            raise AppError(
                CodigoError.PROVIDER_FALLO,
                f"OpenAI SDK lanzó {type(error).__name__}",
            ) from error

        tool_calls = _parse_tool_calls(response.output)
        provider_items = tuple(
            _PROVIDER_OBJECT.validate_python(item.model_dump(mode="json", exclude_none=True))
            for item in response.output
        )
        parts = _build_parts(response.output_text, tool_calls)
        turn = Turn(
            role="assistant",
            parts=parts,
            provider_state=ProviderState(provider="openai", items=provider_items),
        )
        structured = _parse_structured(response.output_text, request.response_model)
        return EngineResult(
            turn=turn,
            tool_calls=tool_calls,
            usage=_parse_usage(response.usage),
            provider="openai",
            model=response.model,
            stop_reason=_stop_reason(response),
            raw=response,
            structured=structured,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._owns_client:
            try:
                async with asyncio.timeout(self._options.timeout_seconds):
                    await self._client.close()
            except TimeoutError as error:
                raise AppError(
                    CodigoError.RUNTIME_CIERRE_TIMEOUT,
                    "OpenAIEngine no cerró su cliente dentro del timeout",
                ) from error
        self._closed = True

    def _create_client(
        self,
        api_key: str | None,
        gateway: OpenAICompatibleGateway | None,
    ) -> AsyncOpenAI:
        if gateway is not None:
            return AsyncOpenAI(
                api_key=gateway.api_key.get_secret_value(),
                base_url=gateway.base_url,
                max_retries=0,
                timeout=self._options.timeout_seconds,
            )
        return AsyncOpenAI(
            api_key=api_key,
            max_retries=self._options.max_retries,
            timeout=self._options.timeout_seconds,
        )

    def _build_payload(self, request: EngineRequest) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "input": _build_input(request.history),
            "instructions": request.instructions,
            "max_output_tokens": self._options.max_output_tokens,
            "store": False,
            "parallel_tool_calls": self._options.parallel_tool_calls,
        }
        if request.tools:
            payload["tools"] = [_tool_payload(spec) for spec in request.tools]
        if request.metadata:
            payload["metadata"] = dict(request.metadata)

        reasoning = self._reasoning_payload()
        if reasoning:
            payload["reasoning"] = reasoning

        text = self._text_payload(request.response_model)
        if text:
            payload["text"] = text
        if self._options.temperature is not None:
            payload["temperature"] = self._options.temperature
        return payload

    def _reasoning_payload(self) -> dict[str, str]:
        reasoning: dict[str, str] = {}
        if self._options.reasoning_effort is not None:
            reasoning["effort"] = self._options.reasoning_effort
        if self._options.reasoning_context is not None:
            reasoning["context"] = self._options.reasoning_context
        return reasoning

    def _text_payload(self, response_model: type[BaseModel] | None) -> dict[str, object]:
        text: dict[str, object] = {}
        if self._options.text_verbosity is not None:
            text["verbosity"] = self._options.text_verbosity
        if response_model is not None:
            text["format"] = {
                "type": "json_schema",
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                "strict": True,
            }
        return text


def _build_input(history: tuple[Turn, ...]) -> list[ResponseInputItemParam]:
    items: list[ResponseInputItemParam] = []
    for turn in history:
        if turn.provider_state is not None:
            if turn.provider_state.provider != "openai":
                raise AppError(
                    CodigoError.PROVIDER_RESPUESTA_INVALIDA,
                    "historial contiene estado de otro provider",
                )
            items.extend(
                cast(ResponseInputItemParam, thaw_provider(cast(FrozenProviderValue, item)))
                for item in turn.provider_state.items
            )
            continue
        if turn.role in {"user", "assistant"}:
            item = {"role": turn.role, "content": turn.text}
            items.append(cast(ResponseInputItemParam, item))
            continue
        for part in turn.parts:
            if not isinstance(part, ToolResultPart):
                continue
            item = {
                "type": "function_call_output",
                "call_id": part.call_id,
                "output": part.output,
            }
            items.append(cast(ResponseInputItemParam, item))
    return items


def _tool_payload(spec: ToolSpec) -> dict[str, object]:
    parameters = {
        key: thaw_json(cast(FrozenJsonValue, value)) for key, value in spec.parameters.items()
    }
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": parameters,
        "strict": spec.strict,
    }


def _parse_tool_calls(output: list[object]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for item in output:
        if not isinstance(item, ResponseFunctionToolCall):
            continue
        try:
            arguments = json.loads(item.arguments)
            arguments = _JSON_OBJECT.validate_python(arguments)
        except (json.JSONDecodeError, ValidationError) as error:
            raise AppError(
                CodigoError.PROVIDER_RESPUESTA_INVALIDA,
                f"argumentos inválidos en call {item.call_id}",
            ) from error
        calls.append(ToolCall(call_id=item.call_id, name=item.name, arguments=arguments))
    return tuple(calls)


def _build_parts(text: str, calls: tuple[ToolCall, ...]):
    parts: list[TextPart | ToolCallPart] = []
    if text:
        parts.append(TextPart(text=text))
    parts.extend(
        ToolCallPart(
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
        )
        for call in calls
    )
    return tuple(parts)


def _parse_structured(text: str, model: type[BaseModel] | None) -> BaseModel | None:
    if model is None or not text:
        return None
    try:
        return model.model_validate_json(text)
    except ValidationError as error:
        raise AppError(
            CodigoError.PROVIDER_RESPUESTA_INVALIDA,
            "structured output no validó contra el contrato",
        ) from error


def _parse_usage(usage) -> Usage:
    if usage is None:
        return Usage()
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
        cached_tokens=getattr(input_details, "cached_tokens", 0) or 0,
        cache_write_tokens=getattr(input_details, "cache_write_tokens", 0) or 0,
    )


def _stop_reason(response) -> str | None:
    incomplete = getattr(response, "incomplete_details", None)
    if incomplete is not None:
        return incomplete.reason
    return response.status
