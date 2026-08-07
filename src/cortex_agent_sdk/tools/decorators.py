from collections.abc import Callable
from typing import overload

from pydantic import BaseModel

from cortex_agent_sdk.tools.models import ToolFunction


@overload
def tool[F: ToolFunction](function: F, /) -> F: ...


@overload
def tool[F: ToolFunction](
    *,
    input_model: type[BaseModel] | None = None,
) -> Callable[[F], F]: ...


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
def final_answer[F: ToolFunction](function: F, /) -> F: ...


@overload
def final_answer[F: ToolFunction](
    *,
    input_model: type[BaseModel] | None = None,
) -> Callable[[F], F]: ...


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
