from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from pydantic import JsonValue

type FrozenJsonValue = (
    bool | int | float | str | tuple[FrozenJsonValue, ...] | Mapping[str, FrozenJsonValue] | None
)
type JsonInputObject = Mapping[str, JsonValue]
type FrozenJsonObject = Mapping[str, FrozenJsonValue]
type JsonObject = JsonInputObject | FrozenJsonObject
type ProviderValue = (
    bool
    | int
    | float
    | str
    | bytes
    | tuple[ProviderValue, ...]
    | Mapping[str, ProviderValue]
    | None
)
type ProviderItem = Mapping[str, ProviderValue]


def freeze_json_object(value: JsonObject) -> FrozenJsonObject:
    frozen = {key: _freeze_json(item) for key, item in value.items()}
    return MappingProxyType(frozen)


def thaw_json_object(value: JsonObject) -> dict[str, JsonValue]:
    return {key: _thaw_json(item) for key, item in value.items()}


def freeze_provider_item(value: ProviderItem) -> ProviderItem:
    frozen = {key: _freeze_provider(item) for key, item in value.items()}
    return MappingProxyType(frozen)


def thaw_provider_item(value: ProviderItem) -> dict[str, object]:
    return {key: _thaw_provider(item) for key, item in value.items()}


def _freeze_json(value: JsonValue | FrozenJsonValue) -> FrozenJsonValue:
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        frozen = {key: _freeze_json(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    return value


def _thaw_json(value: JsonValue | FrozenJsonValue) -> JsonValue:
    if isinstance(value, list | tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    return cast(JsonValue, value)


def _freeze_provider(value: ProviderValue) -> ProviderValue:
    if isinstance(value, tuple):
        return tuple(_freeze_provider(item) for item in value)
    if isinstance(value, Mapping):
        frozen = {key: _freeze_provider(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    return value


def _thaw_provider(value: ProviderValue) -> object:
    if isinstance(value, tuple):
        return [_thaw_provider(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw_provider(item) for key, item in value.items()}
    return value
