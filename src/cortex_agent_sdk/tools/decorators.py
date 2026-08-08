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
    """Construye el decorador interno sin alterar la firma del callable original."""

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
    """Expone una función async como tool normal de Cortex.

    Un resultado exitoso vuelve al modelo y el loop puede seguir razonando o llamar más tools.
    ``input_model`` permite usar un ``BaseModel`` como contrato explícito de entrada.
    ``needs_approval`` puede ser ``True`` o un predicate sync/async evaluado por cada tool call.
    El decorador conserva la firma tipada del callable original.
    """
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
    """Expone una tool cuyo resultado puede actuar como respuesta de respaldo.

    La ejecución no termina el run: el resultado vuelve al modelo igual que en ``@tool``. Cortex
    conserva el último fallback exitoso y sólo lo utiliza si el modelo termina limpiamente sin texto
    visible. No sustituye errores, límites del runtime ni respuestas estructuradas.
    ``input_model`` y ``needs_approval`` tienen la misma semántica que en ``@tool``.
    """
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
    """Expone una tool terminal cuyo ``str`` exitoso se convierte en la respuesta final.

    A diferencia de ``@tool`` y ``@fallback_answer``, una ejecución exitosa corta el loop sin pedir
    otro turno al modelo. La función debe devolver ``str``. Si requiere aprobación, Cortex pausa
    antes de invocarla y conserva el hard stop sólo cuando la call aprobada termina correctamente.
    """
    decorator = _decorator(
        input_model=input_model,
        result_mode=ToolResultMode.FINAL,
        needs_approval=needs_approval,
    )

    if function is None:
        return decorator

    return decorator(function)
