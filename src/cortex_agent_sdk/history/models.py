import base64
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from cortex_agent_sdk.immutable import (
    JsonObject,
    ProviderItem,
    ProviderValue,
    freeze_json_object,
    freeze_provider_item,
    thaw_json_object,
)

_BYTES_TAG = "$cortex.bytes"


class ProviderState(BaseModel):
    """Items nativos necesarios para continuar sin pérdida con un provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    items: tuple[ProviderItem, ...]

    @field_validator("items", mode="before")
    @classmethod
    def decode_bytes(cls, value: object) -> object:
        return _decode_provider_bytes(value)

    @field_validator("items", mode="after")
    @classmethod
    def freeze_items(
        cls,
        value: tuple[ProviderItem, ...],
    ) -> tuple[ProviderItem, ...]:
        return tuple(freeze_provider_item(item) for item in value)

    @field_serializer("items", when_used="json")
    def encode_bytes(
        self,
        items: tuple[ProviderItem, ...],
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


class ToolResultPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    output: str
    is_error: bool = False
    error_code: str | None = None


HistoryPart = Annotated[TextPart | ToolCallPart | ToolResultPart, Field(discriminator="type")]


class Turn(BaseModel):
    """Turno lógico persistible con extensión nativa opcional."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant", "tool"]
    parts: tuple[HistoryPart, ...]
    provider_state: ProviderState | None = None

    @model_validator(mode="after")
    def validate_role_contract(self) -> Self:
        if self.role == "user":
            valid_parts = all(isinstance(part, TextPart) for part in self.parts)
        elif self.role == "assistant":
            valid_parts = all(isinstance(part, TextPart | ToolCallPart) for part in self.parts)
        else:
            valid_parts = all(isinstance(part, ToolResultPart) for part in self.parts)

        if not valid_parts:
            raise ValueError(f"parts incompatibles con role={self.role}")
        if self.role != "assistant" and self.provider_state is not None:
            raise ValueError("provider_state requiere role=assistant")
        return self

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
