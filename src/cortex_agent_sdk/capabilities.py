"""Capacidades opcionales para agentes Pydantic AI."""

from collections.abc import Collection
from dataclasses import replace
from typing import Any

from pydantic_ai import ModelResponse, RunContext, TextPart
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models import ModelRequestContext

__all__ = ["last_tool_result_fallback"]


def last_tool_result_fallback(tool_names: str | Collection[str]) -> Hooks[Any]:
    """Usa el último resultado elegible si el modelo termina sin texto."""
    if isinstance(tool_names, str):
        eligible_tools = frozenset((tool_names,))
    else:
        eligible_tools = frozenset(tool_names)

    def use_last_tool_result(
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if response.finish_reason not in {None, "stop"}:
            return response
        if response.text and response.text.strip():
            return response
        if response.tool_calls:
            return response

        result = _last_tool_result(request_context, ctx.run_id, eligible_tools)
        if result is None:
            return response
        return replace(response, parts=[*response.parts, TextPart(result)])

    return Hooks(after_model_request=use_last_tool_result)


def _last_tool_result(
    request_context: ModelRequestContext,
    run_id: str | None,
    tool_names: Collection[str],
) -> str | None:
    if run_id is None:
        return None
    for message in reversed(request_context.messages):
        if not isinstance(message, ModelRequest) or message.run_id != run_id:
            continue
        for part in reversed(message.parts):
            if (
                isinstance(part, ToolReturnPart)
                and part.outcome == "success"
                and part.tool_name in tool_names
                and isinstance(part.content, str)
                and part.content.strip()
            ):
                return part.content
    return None
