import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Never

from cortex_agent_sdk.engine import EngineRequest, EngineResult, ToolCall
from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.results import AgentResult
from cortex_agent_sdk.tools.execution import ToolOutcome

BeforeModelHook = Callable[[EngineRequest], Awaitable[EngineRequest | None]]
AfterModelHook = Callable[[EngineResult], Awaitable[EngineResult | None]]
BeforeToolHook = Callable[[ToolCall], Awaitable[ToolCall | None]]
AfterToolHook = Callable[[ToolOutcome], Awaitable[ToolOutcome | None]]
TurnFinishedHook = Callable[[AgentResult], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AgentHooks:
    before_model: tuple[BeforeModelHook, ...] = ()
    after_model: tuple[AfterModelHook, ...] = ()
    before_tool: tuple[BeforeToolHook, ...] = ()
    after_tool: tuple[AfterToolHook, ...] = ()
    turn_finished: tuple[TurnFinishedHook, ...] = ()


class HookChain:
    def __init__(self, hooks: AgentHooks, timeout_seconds: float) -> None:
        self._hooks = hooks
        self._timeout_seconds = timeout_seconds

    async def run_before_model(self, request: EngineRequest) -> EngineRequest:
        current = request
        for hook in self._hooks.before_model:
            replacement = await self._call(hook(current), "before_model")
            if replacement is not None:
                if not isinstance(replacement, EngineRequest):
                    self._invalid("before_model")
                try:
                    current = replace(replacement)
                except (TypeError, ValueError):
                    self._invalid("before_model")
        return current

    async def run_after_model(self, result: EngineResult) -> EngineResult:
        current = result
        for hook in self._hooks.after_model:
            replacement = await self._call(hook(current), "after_model")
            if replacement is not None:
                if not isinstance(replacement, EngineResult):
                    self._invalid("after_model")
                try:
                    current = replace(replacement)
                except (TypeError, ValueError):
                    self._invalid("after_model")
        return current

    async def run_before_tool(self, call: ToolCall) -> ToolCall:
        current = call
        for hook in self._hooks.before_tool:
            replacement = await self._call(hook(current), "before_tool")
            if replacement is not None:
                if not isinstance(replacement, ToolCall):
                    self._invalid("before_tool")
                try:
                    payload = replacement.model_dump(mode="json", round_trip=True)
                    current = ToolCall.model_validate(payload)
                except ValueError:
                    self._invalid("before_tool")
        return current

    async def run_after_tool(self, outcome: ToolOutcome) -> ToolOutcome:
        current = outcome
        for hook in self._hooks.after_tool:
            replacement = await self._call(hook(current), "after_tool")
            if replacement is not None:
                if not isinstance(replacement, ToolOutcome):
                    self._invalid("after_tool")
                try:
                    current = replace(replacement)
                except (TypeError, ValueError):
                    self._invalid("after_tool")
        return current

    async def run_turn_finished(self, result: AgentResult) -> None:
        for hook in self._hooks.turn_finished:
            await self._call(hook(result), "turn_finished")

    async def _call(self, operation: Awaitable[object], name: str) -> object:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await operation
        except TimeoutError as error:
            raise AppError(CodigoError.HOOK_FALLO, f"timeout en {name}") from error
        except Exception as error:
            raise AppError(
                CodigoError.HOOK_FALLO,
                f"{name} lanzó {type(error).__name__}",
            ) from error

    def _invalid(self, name: str) -> Never:
        raise AppError(CodigoError.HOOK_FALLO, f"retorno inválido en {name}")
