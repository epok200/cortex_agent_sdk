from typing import Protocol, cast, overload

from cortex_agent_sdk.tools.approval import ToolApprovalRule
from cortex_agent_sdk.tools.models import (
    ToolDecorator,
    ToolDecoratorResult,
    ToolFunction,
    ToolInputModel,
    ToolResultMode,
)


class _ToolMetadata(Protocol):
    """Metadata interna que Cortex añade a una función decorada."""

    __cortex_tool__: bool
    __cortex_input_model__: ToolInputModel
    __cortex_result_mode__: ToolResultMode
    __cortex_needs_approval__: ToolApprovalRule


def _decorator(
    *,
    input_model: ToolInputModel | None,
    result_mode: ToolResultMode,
    needs_approval: ToolApprovalRule,
) -> ToolDecorator:
    def decorate[TTool: ToolFunction](function: TTool) -> TTool:
        metadata = cast(_ToolMetadata, function)
        metadata.__cortex_tool__ = True
        metadata.__cortex_result_mode__ = result_mode
        metadata.__cortex_needs_approval__ = needs_approval

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
    needs_approval: ToolApprovalRule = False,
) -> ToolDecorator: ...


def tool[TTool: ToolFunction](
    function: TTool | None = None,
    *,
    input_model: ToolInputModel | None = None,
    needs_approval: ToolApprovalRule = False,
) -> ToolDecoratorResult[TTool]:
    """Marca una tool que devuelve su resultado al modelo y continúa el loop."""
    decorator = _decorator(
        input_model=input_model,
        result_mode=ToolResultMode.CONTINUE,
        needs_approval=needs_approval,
    )

    if function is None:
        return decorator

    return decorator(function)


@overload
def fallback_answer[TTool: ToolFunction](function: TTool, /) -> TTool: ...


@overload
def fallback_answer(
    *,
    input_model: ToolInputModel | None = None,
    needs_approval: ToolApprovalRule = False,
) -> ToolDecorator: ...


def fallback_answer[TTool: ToolFunction](
    function: TTool | None = None,
    *,
    input_model: ToolInputModel | None = None,
    needs_approval: ToolApprovalRule = False,
) -> ToolDecoratorResult[TTool]:
    """Marca una tool cuyo último resultado puede rescatar un cierre limpio sin texto."""
    decorator = _decorator(
        input_model=input_model,
        result_mode=ToolResultMode.FALLBACK,
        needs_approval=needs_approval,
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
    needs_approval: ToolApprovalRule = False,
) -> ToolDecorator: ...


def final_answer[TTool: ToolFunction](
    function: TTool | None = None,
    *,
    input_model: ToolInputModel | None = None,
    needs_approval: ToolApprovalRule = False,
) -> ToolDecoratorResult[TTool]:
    """Marca una tool cuyo resultado exitoso termina el run inmediatamente."""
    decorator = _decorator(
        input_model=input_model,
        result_mode=ToolResultMode.FINAL,
        needs_approval=needs_approval,
    )

    if function is None:
        return decorator

    return decorator(function)
