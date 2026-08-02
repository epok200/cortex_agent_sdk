import base64
from collections.abc import Mapping
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_serializer, field_validator

from cortex_agent_sdk.immutable import (
    FrozenJsonValue,
    FrozenProviderValue,
    freeze_json,
    freeze_provider,
    thaw_json,
)

_BYTES_TAG = "$cortex.bytes"

ProviderValue = FrozenProviderValue


class ProviderState(BaseModel):
    """Items nativos necesarios para continuar sin pérdida con un provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    items: tuple[Mapping[str, ProviderValue], ...]

    @field_validator("items", mode="before")
    @classmethod
    def decode_bytes(cls, value: object) -> object:
        return _decode_provider_bytes(value)

    @field_validator("items", mode="after")
    @classmethod
    def freeze_items(
        cls,
        value: tuple[Mapping[str, ProviderValue], ...],
    ) -> tuple[Mapping[str, ProviderValue], ...]:
        return tuple(_freeze_provider_item(item) for item in value)

    @field_serializer("items", when_used="json")
    def encode_bytes(
        self,
        items: tuple[Mapping[str, ProviderValue], ...],
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            {key: _encode_provider_bytes(value) for key, value in item.items()} for item in items
        )


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["text"] = "text"
    text: str


class ToolCallPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: Mapping[str, JsonValue]

    @field_validator("arguments", mode="after")
    @classmethod
    def freeze_arguments(
        cls,
        value: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        frozen = freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise TypeError("arguments debe ser un objeto")
        return cast(Mapping[str, JsonValue], frozen)

    @field_serializer("arguments", when_used="json")
    def serialize_arguments(
        self,
        value: Mapping[str, JsonValue],
    ) -> dict[str, object]:
        return {key: thaw_json(cast(FrozenJsonValue, item)) for key, item in value.items()}


class ToolResultPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    output: str
    is_error: bool = False
    error_code: str | None = None


Part = Annotated[TextPart | ToolCallPart | ToolResultPart, Field(discriminator="type")]


class Turn(BaseModel):
    """Turno lógico persistible con extensión nativa opcional."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant", "tool"]
    parts: tuple[Part, ...]
    provider_state: ProviderState | None = None

    @classmethod
    def user(cls, text: str) -> "Turn":
        return cls(role="user", parts=(TextPart(text=text),))

    @property
    def text(self) -> str:
        return "\n".join(part.text for part in self.parts if isinstance(part, TextPart))


def _encode_provider_bytes(value: ProviderValue) -> object:
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return {_BYTES_TAG: encoded}
    if isinstance(value, tuple):
        return [_encode_provider_bytes(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _encode_provider_bytes(item) for key, item in value.items()}
    return value


def _decode_provider_bytes(value: object) -> object:
    if isinstance(value, list):
        return [_decode_provider_bytes(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {_BYTES_TAG} and isinstance(value[_BYTES_TAG], str):
        return base64.b64decode(value[_BYTES_TAG], validate=True)
    return {key: _decode_provider_bytes(item) for key, item in value.items()}


def _freeze_provider_item(
    value: Mapping[str, ProviderValue],
) -> Mapping[str, ProviderValue]:
    frozen = freeze_provider(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("provider item debe ser un objeto")
    return frozen

