from typing import Protocol, cast, overload

from cortex_agent_sdk.tools.models import (
    ToolDecorator,
    ToolDecoratorResult,
    ToolFunction,
    ToolInputModel,
)


class _ToolMetadata(Protocol):
    """Metadata interna que Cortex añade a una función decorada."""

    __cortex_tool__: bool
    __cortex_final_answer__: bool
    __cortex_input_model__: ToolInputModel


def _decorator(
    *,
    input_model: ToolInputModel | None,
    is_final_answer: bool,
) -> ToolDecorator:
    def decorate[TTool: ToolFunction](function: TTool) -> TTool:
        metadata = cast(_ToolMetadata, function)

        metadata.__cortex_tool__ = True

        if is_final_answer:
            metadata.__cortex_final_answer__ = True

        if input_model is not None:
            metadata.__cortex_input_model__ = input_model

        return function

    return decorate


@overload
def tool[TTool: ToolFunction](function: TTool, /) -> TTool: ...


@overload
def tool(
    *,
    input_model: ToolInputModel | None = None,
) -> ToolDecorator: ...


def tool[TTool: ToolFunction](
    function: TTool | None = None,
    *,
    input_model: ToolInputModel | None = None,
) -> ToolDecoratorResult[TTool]:
    """Marca una tool y permite declarar un contrato Pydantic explícito."""
    decorator = _decorator(
        input_model=input_model,
        is_final_answer=False,
    )

    if function is None:
        return decorator

    return decorator(function)


@overload
def final_answer[TTool: ToolFunction](function: TTool, /) -> TTool: ...


@overload
def final_answer(
    *,
    input_model: ToolInputModel | None = None,
) -> ToolDecorator: ...


def final_answer[TTool: ToolFunction](
    function: TTool | None = None,
    *,
    input_model: ToolInputModel | None = None,
) -> ToolDecoratorResult[TTool]:
    """Marca una tool terminal y permite un contrato Pydantic explícito."""
    decorator = _decorator(
        input_model=input_model,
        is_final_answer=True,
    )

    if function is None:
        return decorator

    return decorator(function)