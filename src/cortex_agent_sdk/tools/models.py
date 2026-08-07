from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from cortex_agent_sdk.immutable import JsonObject, freeze_json_object, thaw_json_object


class _InjectedMarker:
    def __repr__(self) -> str:
        return "Injected"


Injected = _InjectedMarker()


class ToolSpec(BaseModel):
    """Schema neutral visible para el modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    parameters: JsonObject
    strict: bool = False

    @field_validator("parameters", mode="after")
    @classmethod
    def freeze_parameters(
        cls,
        value: JsonObject,
    ) -> JsonObject:
        return freeze_json_object(value)

    @field_serializer("parameters", when_used="json")
    def serialize_parameters(
        self,
        value: JsonObject,
    ) -> dict[str, JsonValue]:
        return thaw_json_object(value)


ToolFunction = Callable[..., Awaitable[object]]
type ToolInputModel = type[BaseModel]


class ToolResultMode(StrEnum):
    """Semántica que el runtime aplica al resultado exitoso de una tool."""

    CONTINUE = "continue"
    FALLBACK = "fallback"
    FINAL = "final"


class ToolDecorator(Protocol):
    """Decorador que conserva exactamente el callable de una tool."""

    def __call__[TTool: ToolFunction](self, function: TTool, /) -> TTool: ...


type ToolDecoratorResult[TTool: ToolFunction] = TTool | ToolDecorator


@dataclass(frozen=True, slots=True)
class ToolBinding:
    """Une callable, contrato opcional y argumentos privados."""

    function: ToolFunction
    spec: ToolSpec | None = None
    private_arguments: Mapping[str, object] = field(default_factory=dict)
    input_model: type[BaseModel] | None = None
    result_mode: ToolResultMode | None = None
    needs_approval: bool | None = None

    def __post_init__(self) -> None:
        frozen_arguments = MappingProxyType(dict(self.private_arguments))
        object.__setattr__(self, "private_arguments", frozen_arguments)
