from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from cortex_agent_sdk.history.models import ToolCallPart, Turn
from cortex_agent_sdk.immutable import JsonObject, freeze_json_object, thaw_json_object
from cortex_agent_sdk.tools.models import ToolSpec


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    name: str
    arguments: JsonObject

    @field_validator("arguments", mode="after")
    @classmethod
    def freeze_arguments(
        cls,
        value: JsonObject,
    ) -> JsonObject:
        return freeze_json_object(value)

    @field_serializer("arguments", when_used="json")
    def serialize_arguments(
        self,
        value: JsonObject,
    ) -> dict[str, JsonValue]:
        return thaw_json_object(value)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.input_tokens,
                self.output_tokens,
                self.reasoning_tokens,
                self.cached_tokens,
                self.cache_write_tokens,
            )
            < 0
        ):
            raise ValueError("usage no admite valores negativos")

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class EngineRequest:
    instructions: str
    history: tuple[Turn, ...]
    tools: tuple[ToolSpec, ...]
    response_model: type[BaseModel] | None = None
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        valid_history = isinstance(self.history, tuple) and all(
            isinstance(turn, Turn) for turn in self.history
        )
        valid_tools = isinstance(self.tools, tuple) and all(
            isinstance(spec, ToolSpec) for spec in self.tools
        )
        valid_model = self.response_model is None or (
            isinstance(self.response_model, type) and issubclass(self.response_model, BaseModel)
        )
        if (
            not isinstance(self.instructions, str)
            or not valid_history
            or not valid_tools
            or not valid_model
        ):
            raise TypeError("EngineRequest inválido")
        if self.metadata is not None:
            if not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in self.metadata.items()
            ):
                raise TypeError("metadata inválida")
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EngineResult:
    turn: Turn
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    provider: str
    model: str
    stop_reason: str | None
    raw: object
    structured: BaseModel | None = None

    def __post_init__(self) -> None:
        valid_calls = isinstance(self.tool_calls, tuple) and all(
            isinstance(call, ToolCall) for call in self.tool_calls
        )
        valid_structured = self.structured is None or isinstance(self.structured, BaseModel)
        if not isinstance(self.turn, Turn) or self.turn.role != "assistant" or not valid_calls:
            raise TypeError("EngineResult inválido")
        turn_calls = tuple(part for part in self.turn.parts if isinstance(part, ToolCallPart))
        aligned_calls = len(turn_calls) == len(self.tool_calls) and all(
            part.call_id == call.call_id
            and part.name == call.name
            and part.arguments == call.arguments
            for part, call in zip(turn_calls, self.tool_calls, strict=True)
        )
        if not aligned_calls:
            raise TypeError("tool_calls no coincide con el turno")
        valid_identity = (
            isinstance(self.provider, str)
            and bool(self.provider)
            and isinstance(self.model, str)
            and bool(self.model)
        )
        provider_state = self.turn.provider_state
        state_matches = provider_state is None or provider_state.provider == self.provider
        if not isinstance(self.usage, Usage) or not valid_identity or not state_matches:
            raise TypeError("EngineResult inválido")
        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise TypeError("stop_reason inválido")
        if not valid_structured:
            raise TypeError("structured inválido")


@runtime_checkable
class ModelEngine(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def generate(self, request: EngineRequest) -> EngineResult: ...

    async def aclose(self) -> None: ...
