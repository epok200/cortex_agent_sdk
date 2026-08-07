from typing import overload

from cortex_agent_sdk.tools.models import (
    ToolDecorator,
    ToolDecoratorResult,
    ToolFunction,
    ToolInputModel,
)


def _decorator(
    *,
    input_model: ToolInputModel | None,
    final_answer: bool,
) -> ToolDecorator:
    def decorate[TTool: ToolFunction](function: TTool) -> TTool:
        setattr(function, "__cortex_tool__", True)

        if final_answer:
            setattr(function, "__cortex_final_answer__", True)

        if input_model is not None:
            setattr(function, "__cortex_input_model__", input_model)

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
    decorator = _decorator(input_model=input_model, final_answer=False)

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
    decorator = _decorator(input_model=input_model, final_answer=True)

    if function is None:
        return decorator

    return decorator(function)
