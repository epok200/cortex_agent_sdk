from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from pydantic import JsonValue

type FrozenJsonValue = (
    bool | int | float | str | tuple[FrozenJsonValue, ...] | Mapping[str, FrozenJsonValue] | None
)
type FrozenProviderValue = (
    bool
    | int
    | float
    | str
    | bytes
    | tuple[FrozenProviderValue, ...]
    | Mapping[str, FrozenProviderValue]
    | None
)


def freeze_json(value: JsonValue | Mapping[str, JsonValue]) -> FrozenJsonValue:
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        frozen = {key: freeze_json(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    return value


def thaw_json(value: FrozenJsonValue) -> JsonValue:
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    return cast(JsonValue, value)


def freeze_provider(value: FrozenProviderValue) -> FrozenProviderValue:
    if isinstance(value, tuple):
        return tuple(freeze_provider(item) for item in value)
    if isinstance(value, Mapping):
        frozen = {key: freeze_provider(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    return value


def thaw_provider(value: FrozenProviderValue) -> object:
    if isinstance(value, tuple):
        return [thaw_provider(item) for item in value]
    if isinstance(value, Mapping):
        return {key: thaw_provider(item) for key, item in value.items()}
    return value

