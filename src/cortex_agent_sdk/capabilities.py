"""Capacidades opcionales para agentes Pydantic AI."""

from collections.abc import Collection
from dataclasses import replace
from typing import Any

from pydantic_ai import CallToolsNode, ModelRequestNode, ModelResponse, RunContext, TextPart
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext

from cortex_agent_sdk.sessions import Session

__all__ = ["last_tool_result_fallback", "session_checkpoints"]


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

    def recover_provider_failure(
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        if not isinstance(error, ModelAPIError):
            raise error
        result = _last_tool_result(request_context, ctx.run_id, eligible_tools)
        if result is None:
            raise error
        return ModelResponse(parts=[TextPart(result)], finish_reason="stop")

    return Hooks(
        after_model_request=use_last_tool_result,
        model_request_error=recover_provider_failure,
    )


def session_checkpoints(
    session: Session,
    *,
    effect_tools: str | Collection[str],
) -> Hooks[Any]:
    """Persiste tools completadas y bloquea efectos ambiguos."""
    if isinstance(effect_tools, str):
        effect_tool_names = frozenset((effect_tools,))
    else:
        effect_tool_names = frozenset(effect_tools)
    pending_runs: set[str | None] = set()

    async def checkpoint_node(
        ctx: RunContext[Any],
        *,
        node: Any,
        result: Any,
    ) -> Any:
        if isinstance(result, CallToolsNode):
            effect_calls = tuple(
                call
                for call in result.model_response.tool_calls
                if call.tool_name in effect_tool_names
            )
            if effect_calls:
                await session.start_tool_turn(effect_calls)
                pending_runs.add(ctx.run_id)

        if isinstance(node, CallToolsNode) and node.model_response.tool_calls:
            effect_calls = tuple(
                call
                for call in node.model_response.tool_calls
                if call.tool_name in effect_tool_names
            )
            if effect_calls and not _effect_calls_succeeded(ctx, result, effect_calls):
                return result
            if effect_calls:
                pending_runs.discard(ctx.run_id)
            if ctx.run_id in pending_runs:
                return result

            messages = list(ctx.messages)
            if isinstance(result, ModelRequestNode):
                messages.append(result.request)
            await session.checkpoint(messages)
            pending_runs.discard(ctx.run_id)

        return result

    return Hooks(after_node_run=checkpoint_node)


def _effect_calls_succeeded(
    ctx: RunContext[Any],
    result: Any,
    effect_calls: Collection[ToolCallPart],
) -> bool:
    if isinstance(result, ModelRequestNode):
        request = result.request
    else:
        request = next(
            (
                message
                for message in reversed(ctx.messages)
                if isinstance(message, ModelRequest) and message.run_id == ctx.run_id
            ),
            None,
        )
    if request is None:
        return False

    successful = {
        (part.tool_name, part.tool_call_id)
        for part in request.parts
        if isinstance(part, ToolReturnPart) and part.outcome == "success"
    }
    return all((call.tool_name, call.tool_call_id) in successful for call in effect_calls)


def _last_tool_result(
    request_context: ModelRequestContext,
    run_id: str | None,
    tool_names: Collection[str],
) -> str | None:
    if run_id is None:
        return None
    candidate: str | None = None
    for message in request_context.messages:
        if not isinstance(message, ModelRequest) or message.run_id != run_id:
            continue
        for part in message.parts:
            if isinstance(part, RetryPromptPart) and part.tool_name is not None:
                return None
            if not isinstance(part, ToolReturnPart):
                continue
            if part.outcome != "success":
                return None
            if (
                part.tool_name in tool_names
                and isinstance(part.content, str)
                and part.content.strip()
            ):
                candidate = part.content
    return candidate
