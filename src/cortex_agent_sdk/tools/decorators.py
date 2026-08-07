from collections.abc import Callable
from typing import TypeVar, overload

from pydantic import BaseModel

from cortex_agent_sdk.tools.models import ToolFunction

_ToolCallable = TypeVar("_ToolCallable", bound=ToolFunction)


@overload
def tool(function: _ToolCallable, /) -> _ToolCallable: ...


@overload
def tool(
    *,
    input_model: type[BaseModel] | None = None,
) -> Callable[[_ToolCallable], _ToolCallable]: ...


def tool(
    function: ToolFunction | None = None,
    *,
    input_model: type[BaseModel] | None = None,
) -> ToolFunction | Callable[[ToolFunction], ToolFunction]:
    """Marca una tool y permite declarar un contrato Pydantic explícito."""

    def decorate(inner: ToolFunction) -> ToolFunction:
        inner.__cortex_tool__ = True  # type: ignore[attr-defined]
        if input_model is not None:
            inner.__cortex_input_model__ = input_model  # type: ignore[attr-defined]
        return inner

    if function is None:
        return decorate
    return decorate(function)


@overload
def final_answer(function: _ToolCallable, /) -> _ToolCallable: ...


@overload
def final_answer(
    *,
    input_model: type[BaseModel] | None = None,
) -> Callable[[_ToolCallable], _ToolCallable]: ...


def final_answer(
    function: ToolFunction | None = None,
    *,
    input_model: type[BaseModel] | None = None,
) -> ToolFunction | Callable[[ToolFunction], ToolFunction]:
    """Marca una tool terminal y permite un contrato Pydantic explícito."""

    def decorate(inner: ToolFunction) -> ToolFunction:
        inner.__cortex_tool__ = True  # type: ignore[attr-defined]
        inner.__cortex_final_answer__ = True  # type: ignore[attr-defined]
        if input_model is not None:
            inner.__cortex_input_model__ = input_model  # type: ignore[attr-defined]
        return inner

    if function is None:
        return decorate
    return decorate(function)
