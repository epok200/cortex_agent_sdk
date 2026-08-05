from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

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


@dataclass(frozen=True, slots=True)
class ToolBinding:
    """Une callable, schema opcional y argumentos privados."""

    function: ToolFunction
    spec: ToolSpec | None = None
    private_arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen_arguments = MappingProxyType(dict(self.private_arguments))
        object.__setattr__(self, "private_arguments", frozen_arguments)
