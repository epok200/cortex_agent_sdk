from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from cortex_agent_sdk.immutable import FrozenJsonValue, freeze_json, thaw_json


class _InjectedMarker:
    def __repr__(self) -> str:
        return "Injected"


Injected = _InjectedMarker()


class ToolSpec(BaseModel):
    """Schema neutral visible para el modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    parameters: Mapping[str, JsonValue]
    strict: bool = False

    @field_validator("parameters", mode="after")
    @classmethod
    def freeze_parameters(
        cls,
        value: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        frozen = freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise TypeError("parameters debe ser un objeto")
        return cast(Mapping[str, JsonValue], frozen)

    @field_serializer("parameters", when_used="json")
    def serialize_parameters(
        self,
        value: Mapping[str, JsonValue],
    ) -> dict[str, object]:
        return {key: thaw_json(cast(FrozenJsonValue, item)) for key, item in value.items()}


ToolFunction = Callable[..., Awaitable[object]]


@dataclass(frozen=True, slots=True)
class ToolBinding:
    """Une callable, schema opcional y argumentos privados."""

    function: ToolFunction
    spec: ToolSpec | None = None
    private_arguments: Mapping[str, object] = field(default_factory=dict)

