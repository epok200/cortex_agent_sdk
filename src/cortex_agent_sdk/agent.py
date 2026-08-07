import asyncio
from collections.abc import Awaitable, Callable, Iterable
from functools import partial
from inspect import isawaitable
from typing import Self
from uuid import uuid4

from pydantic import BaseModel

from cortex_agent_sdk.control import (
    ApprovalAction,
    ApprovalDecision,
    ContinueRun,
    FinalOutput,
    PendingRun,
    ToolApproval,
    ToolResultContext,
    ToolResultPolicy,
)
from cortex_agent_sdk.engine import EngineRequest, EngineResult, ModelEngine, ToolCall
from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.history.models import Turn
from cortex_agent_sdk.history.pipeline import HistoryPipeline
from cortex_agent_sdk.history.transform import HistoryTransform
from cortex_agent_sdk.hooks import AgentHooks, HookChain
from cortex_agent_sdk.lifecycle import AgentLifecycle
from cortex_agent_sdk.results import AgentExitReason, AgentResult
from cortex_agent_sdk.runtime import AgentOptions, RunState
from cortex_agent_sdk.sessions import MemorySessionStore, SessionRecord, SessionStore
from cortex_agent_sdk.sessions.lease import LeaseKeeper
from cortex_agent_sdk.tools.execution import ToolExecutor, ToolOutcome, ToolSet
from cortex_agent_sdk.tools.models import ToolBinding, ToolFunction


class Agent:
    """Fachada async y dueño único del loop de tools."""

    def __init__(
        self,
        engine: ModelEngine,
        *,
        instructions: str = "",
        tools: Iterable[ToolFunction | ToolBinding] = (),
        hooks: AgentHooks | None = None,
        history_transform: HistoryTransform | None = None,
        session_store: SessionStore | None = None,
        options: AgentOptions | None = None,
        own_engine: bool = True,
        own_session_store: bool = False,
        tool_result_policy: ToolResultPolicy | None = None,
    ) -> None:
        self._engine = engine
        self._instructions = instructions
        self._static_tools = tuple(tools)
        self._options = options or AgentOptions()
        self._history = HistoryPipeline(history_transform, self._options.max_history_turns)
        self._hooks = HookChain(hooks or AgentHooks(), self._options.hook_timeout_seconds)
        if session_store is None:
            self._session_store = MemorySessionStore()
        else:
            self._session_store = session_store
        self._own_engine = own_engine
        self._own_session_store = session_store is None or own_session_store
        self._tool_result_policy = tool_result_policy
        self._lifecycle = AgentLifecycle()

    async def __aenter__(self) -> Self:
        await self._lifecycle.ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def run(
        self,
        text: str,
        *,
        session_id: str | None = None,
        tools: Iterable[ToolFunction | ToolBinding] = (),
        response_model: type[BaseModel] | None = None,
        instructions: str | None = None,
    ) -> AgentResult:
        async with self._lifecycle.run():
            if not text.strip():
                raise AppError(CodigoError.CONFIG_INVALIDA, "text no puede estar vacío")

            tool_set = ToolSet.build((*self._static_tools, *tuple(tools)))
            run_instructions = self._instructions if instructions is None else instructions
            if session_id is None:
                state = RunState(
                    history=[Turn.user(text)],
                    record=None,
                    lease_keeper=None,
                    instructions=run_instructions,
                )
                return await self._run_loop(state, tool_set, response_model, None)

            timeout = self._options.session_lock_timeout_seconds
            async with self._session_store.acquire(session_id, timeout) as lease:
                lease_keeper = LeaseKeeper(lease)
                await lease_keeper.start()
                try:
                    record = await lease_keeper.load()
                    if record is None:
                        record = SessionRecord.create(
                            session_id, self._engine.provider, self._engine.model
                        )
                    self._validate_record(record, session_id)
                    history = [*record.history, Turn.user(text)]
                    state = RunState(
                        history=history,
                        record=record,
                        lease_keeper=lease_keeper,
                        instructions=run_instructions,
                    )
                    return await self._run_loop(state, tool_set, response_model, session_id)
                finally:
                    await lease_keeper.aclose()

    async def resume(
        self,
        pending_run: PendingRun,
        *,
        tools: Iterable[ToolFunction | ToolBinding] = (),
        response_model: type[BaseModel] | None = None,
    ) -> AgentResult:
        """Reanuda un run pausado después de resolver sus approvals."""
        async with self._lifecycle.run():
            self._validate_pending_identity(pending_run)
            self._validate_response_model(pending_run, response_model)
            if pending_run.unresolved:
                return self._pending_result_from_snapshot(pending_run)

            tool_set = ToolSet.build((*self._static_tools, *tuple(tools)))
            if pending_run.session_id is None:
                state = self._state_from_pending(pending_run, None, None)
                return await self._resume_pending(
                    state,
                    pending_run,
                    tool_set,
                    response_model,
                )

            session_id = pending_run.session_id
            timeout = self._options.session_lock_timeout_seconds
            async with self._session_store.acquire(session_id, timeout) as lease:
                lease_keeper = LeaseKeeper(lease)
                await lease_keeper.start()
                try:
                    record = await lease_keeper.load()
                    if record is None:
                        raise AppError(
                            CodigoError.SESION_INVALIDA,
                            f"no existe la sesión pausada {session_id}",
                        )
                    self._validate_pending_record(record, pending_run)
                    state = self._state_from_pending(pending_run, record, lease_keeper)
                    return await self._resume_pending(
                        state,
                        pending_run,
                        tool_set,
                        response_model,
                    )
                finally:
                    await lease_keeper.aclose()

    async def reset_session(self, session_id: str) -> None:
        async with self._lifecycle.run():
            if not session_id:
                raise AppError(CodigoError.CONFIG_INVALIDA, "session_id no puede estar vacío")
            await self._session_store.reset(
                session_id,
                self._options.session_lock_timeout_seconds,
            )

    async def aclose(self) -> None:
        await self._lifecycle.close(
            self._close_resources,
            self._options.shutdown_timeout_seconds,
        )

    async def _close_resources(self) -> None:
        try:
            if self._own_engine:
                await self._engine.aclose()
        finally:
            if self._own_session_store:
                await self._session_store.aclose()

    async def _run_loop(
        self,
        state: RunState,
        tool_set: ToolSet,
        response_model: type[BaseModel] | None,
        session_id: str | None,
        *,
        start_step: int = 1,
    ) -> AgentResult:
        executor = ToolExecutor(tool_set, self._options.tool_timeout_seconds)
        for step in range(start_step, self._options.max_steps + 1):
            engine_result = await self._generate(state, tool_set, response_model, session_id)
            self._record_engine_result(state, engine_result)

            if not engine_result.tool_calls:
                if self._should_use_fallback(state, engine_result, response_model):
                    state.last_text = state.fallback_text
                    return await self._finish(state, AgentExitReason.FALLBACK_ANSWER, step)
                return await self._finish(state, AgentExitReason.COMPLETED, step)

            self._validate_tool_batch(tool_set, engine_result.tool_calls)
            await self._start_tool_calls(state, engine_result.tool_calls)
            interruptions = self._approval_requests(tool_set, engine_result.tool_calls)
            if interruptions:
                pending = self._build_pending_run(
                    state,
                    engine_result.tool_calls,
                    interruptions,
                    response_model,
                    session_id,
                    step,
                )
                return self._pause_for_approval(state, pending)

            outcomes = await self._execute_tool_calls(state, executor, engine_result.tool_calls)
            await self._finish_tool_calls(state)
            finished = await self._after_tool_batch(state, outcomes, step, session_id)
            if finished is not None:
                return finished

        return await self._finish(state, AgentExitReason.MAX_STEPS, self._options.max_steps)

    async def _resume_pending(
        self,
        state: RunState,
        pending_run: PendingRun,
        tool_set: ToolSet,
        response_model: type[BaseModel] | None,
    ) -> AgentResult:
        self._validate_tool_batch(tool_set, pending_run.calls)
        self._validate_pending_tools(tool_set, pending_run)
        executor = ToolExecutor(tool_set, self._options.tool_timeout_seconds)
        outcomes = await self._execute_tool_calls(
            state,
            executor,
            pending_run.calls,
            decisions=pending_run.decisions,
        )
        await self._finish_tool_calls(state)
        finished = await self._after_tool_batch(
            state,
            outcomes,
            pending_run.step,
            pending_run.session_id,
        )
        if finished is not None:
            return finished
        if pending_run.step >= self._options.max_steps:
            return await self._finish(
                state,
                AgentExitReason.MAX_STEPS,
                self._options.max_steps,
            )
        return await self._run_loop(
            state,
            tool_set,
            response_model,
            pending_run.session_id,
            start_step=pending_run.step + 1,
        )

    async def _after_tool_batch(
        self,
        state: RunState,
        outcomes: tuple[ToolOutcome, ...],
        step: int,
        session_id: str | None,
    ) -> AgentResult | None:
        if state.consecutive_tool_failures >= self._options.max_consecutive_tool_failures:
            return await self._finish(state, AgentExitReason.TOOL_FAILURE_LIMIT, step)

        for outcome in outcomes:
            if outcome.fallback_answer and not outcome.failed:
                state.fallback_text = outcome.output

        all_final = outcomes and all(
            outcome.final_answer and not outcome.failed for outcome in outcomes
        )
        if all_final:
            state.last_text = "\n\n".join(outcome.output for outcome in outcomes)
            return await self._finish(state, AgentExitReason.FINAL_ANSWER, step)

        policy_result = await self._run_tool_result_policy(state, outcomes, step, session_id)
        if isinstance(policy_result, FinalOutput):
            state.last_text = policy_result.output
            return await self._finish(state, AgentExitReason.TOOL_POLICY, step)
        return None

    async def _run_tool_result_policy(
        self,
        state: RunState,
        outcomes: tuple[ToolOutcome, ...],
        step: int,
        session_id: str | None,
    ) -> ContinueRun | FinalOutput:
        if self._tool_result_policy is None:
            return ContinueRun()

        context = ToolResultContext(
            step=step,
            session_id=session_id,
            history=tuple(state.history),
            outcomes=outcomes,
        )
        decision = self._tool_result_policy(context)
        if isawaitable(decision):
            decision = await decision
        if not isinstance(decision, ContinueRun | FinalOutput):
            raise AppError(
                CodigoError.CONFIG_INVALIDA,
                "tool_result_policy devolvió una decisión inválida",
            )
        return decision

    async def _generate(
        self,
        state: RunState,
        tool_set: ToolSet,
        response_model: type[BaseModel] | None,
        session_id: str | None,
    ) -> EngineResult:
        history = await self._history.prepare(tuple(state.history), session_id)
        request = EngineRequest(
            instructions=state.instructions,
            history=history,
            tools=tool_set.specs,
            response_model=response_model,
        )
        request = await self._hooks.run_before_model(request)
        try:
            async with asyncio.timeout(self._options.provider_timeout_seconds):
                result = await self._with_lease(
                    state,
                    lambda: self._engine.generate(request),
                )
        except TimeoutError as error:
            raise AppError(CodigoError.PROVIDER_TIMEOUT, "provider agotó el timeout") from error
        result = await self._hooks.run_after_model(result)
        if result.provider != self._engine.provider:
            raise AppError(CodigoError.PROVIDER_RESPUESTA_INVALIDA, "provider efectivo inválido")
        return result

    def _record_engine_result(self, state: RunState, result: EngineResult) -> None:
        state.history.append(result.turn)
        state.usage = state.usage + result.usage
        state.raw_responses.append(result.raw)
        state.last_text = result.turn.text or state.last_text
        state.structured = result.structured or state.structured
        state.effective_model = result.model

    def _should_use_fallback(
        self,
        state: RunState,
        result: EngineResult,
        response_model: type[BaseModel] | None,
    ) -> bool:
        clean_stop = result.stop_reason in {None, "completed", "stop"}
        return (
            response_model is None
            and state.fallback_text is not None
            and not result.turn.text
            and clean_stop
        )

    def _approval_requests(
        self,
        tool_set: ToolSet,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolApproval, ...]:
        return tuple(
            ToolApproval.from_call(call) for call in calls if tool_set.needs_approval(call.name)
        )

    def _build_pending_run(
        self,
        state: RunState,
        calls: tuple[ToolCall, ...],
        interruptions: tuple[ToolApproval, ...],
        response_model: type[BaseModel] | None,
        session_id: str | None,
        step: int,
    ) -> PendingRun:
        return PendingRun(
            provider=self._engine.provider,
            model=self._engine.model,
            instructions=state.instructions,
            history=tuple(state.history),
            step=step,
            usage=state.usage,
            tool_calls=state.tool_calls,
            consecutive_tool_failures=state.consecutive_tool_failures,
            last_text=state.last_text,
            fallback_text=state.fallback_text,
            effective_model=state.effective_model,
            session_id=session_id,
            calls=calls,
            interruptions=interruptions,
            response_model_name=_response_model_name(response_model),
        )

    def _pause_for_approval(self, state: RunState, pending_run: PendingRun) -> AgentResult:
        return AgentResult(
            text=state.last_text,
            reason=AgentExitReason.APPROVAL_REQUIRED,
            usage=state.usage,
            steps=pending_run.step,
            tool_calls=state.tool_calls,
            provider=self._engine.provider,
            model=state.effective_model or self._engine.model,
            history=tuple(state.history),
            raw_responses=tuple(state.raw_responses),
            structured=state.structured,
            interruptions=pending_run.unresolved,
            pending_run=pending_run,
        )

    def _pending_result_from_snapshot(self, pending_run: PendingRun) -> AgentResult:
        return AgentResult(
            text=pending_run.last_text,
            reason=AgentExitReason.APPROVAL_REQUIRED,
            usage=pending_run.usage,
            steps=pending_run.step,
            tool_calls=pending_run.tool_calls,
            provider=pending_run.provider,
            model=pending_run.effective_model or pending_run.model,
            history=pending_run.history,
            raw_responses=(),
            interruptions=pending_run.unresolved,
            pending_run=pending_run,
        )

    def _state_from_pending(
        self,
        pending_run: PendingRun,
        record: SessionRecord | None,
        lease_keeper: LeaseKeeper | None,
    ) -> RunState:
        return RunState(
            history=list(pending_run.history),
            record=record,
            lease_keeper=lease_keeper,
            instructions=pending_run.instructions,
            usage=pending_run.usage,
            tool_calls=pending_run.tool_calls,
            consecutive_tool_failures=pending_run.consecutive_tool_failures,
            last_text=pending_run.last_text,
            fallback_text=pending_run.fallback_text,
            effective_model=pending_run.effective_model,
        )

    async def _start_tool_calls(
        self,
        state: RunState,
        calls: tuple[ToolCall, ...],
    ) -> None:
        if state.record is None:
            return
        call_ids = tuple(call.call_id for call in calls)
        state.record = state.record.start_turn(uuid4().hex, call_ids, tuple(state.history))
        await self._save_record(state)

    async def _execute_tool_calls(
        self,
        state: RunState,
        executor: ToolExecutor,
        calls: tuple[ToolCall, ...],
        *,
        decisions: dict[str, ApprovalDecision] | None = None,
    ) -> tuple[ToolOutcome, ...]:
        outcomes: list[ToolOutcome] = []
        for original_call in calls:
            decision = None if decisions is None else decisions.get(original_call.call_id)
            if decision is not None and decision.action is ApprovalAction.REJECT:
                message = decision.rejection_message or "La ejecución fue rechazada por el usuario."
                outcome = ToolOutcome.rejected(original_call, message)
                await self._checkpoint_outcome(state, outcome)
                outcomes.append(outcome)
                continue

            if state.consecutive_tool_failures >= self._options.max_consecutive_tool_failures:
                outcome = ToolOutcome.skipped(original_call)
                await self._checkpoint_outcome(state, outcome)
                outcomes.append(outcome)
                continue

            call = await self._hooks.run_before_tool(original_call)
            if call.call_id != original_call.call_id or call.name != original_call.name:
                raise AppError(CodigoError.HOOK_FALLO, "before_tool cambió identidad de la call")
            original_outcome = await self._with_lease(
                state,
                partial(executor.execute, call),
            )
            try:
                outcome = await self._hooks.run_after_tool(original_outcome)
            except AppError:
                await self._checkpoint_outcome(state, original_outcome)
                raise
            if outcome.call != call:
                await self._checkpoint_outcome(state, original_outcome)
                raise AppError(CodigoError.HOOK_FALLO, "after_tool cambió identidad de la call")
            await self._checkpoint_outcome(state, outcome)
            outcomes.append(outcome)
            state.tool_calls += 1
            if outcome.failed:
                state.consecutive_tool_failures += 1
            else:
                state.consecutive_tool_failures = 0
        return tuple(outcomes)

    async def _checkpoint_outcome(self, state: RunState, outcome: ToolOutcome) -> None:
        turn = Turn(role="tool", parts=(outcome.as_history_part(),))
        state.history.append(turn)
        if state.record is None:
            return
        history = tuple(state.history)
        state.record = state.record.complete_call(outcome.call.call_id, history)
        await self._save_record(state)

    async def _finish_tool_calls(self, state: RunState) -> None:
        if state.record is None:
            return
        history = self._history.window(tuple(state.history))
        state.history = list(history)
        state.record = state.record.finish_turn(history)
        await self._save_record(state)

    async def _finish(
        self,
        state: RunState,
        reason: AgentExitReason,
        steps: int,
    ) -> AgentResult:
        state.history = list(self._history.window(tuple(state.history)))
        result = AgentResult(
            text=state.last_text,
            reason=reason,
            usage=state.usage,
            steps=steps,
            tool_calls=state.tool_calls,
            provider=self._engine.provider,
            model=state.effective_model or self._engine.model,
            history=tuple(state.history),
            raw_responses=tuple(state.raw_responses),
            structured=state.structured,
        )
        await self._hooks.run_turn_finished(result)
        if state.record is not None:
            state.record = state.record.finish_turn(tuple(state.history))
            await self._save_record(state)
        return result

    def _validate_tool_batch(self, tool_set: ToolSet, calls: tuple[ToolCall, ...]) -> None:
        final_calls = [call for call in calls if tool_set.is_final_answer(call.name)]
        if final_calls and len(calls) > 1:
            raise AppError(
                CodigoError.PROVIDER_RESPUESTA_INVALIDA,
                "provider mezcló una tool terminal con otras calls en el mismo paso",
            )

    def _validate_pending_identity(self, pending_run: PendingRun) -> None:
        if pending_run.provider != self._engine.provider or pending_run.model != self._engine.model:
            raise AppError(
                CodigoError.SESION_ENGINE_DISTINTO,
                f"run ligado a {pending_run.provider}/{pending_run.model}",
            )
        if pending_run.step > self._options.max_steps:
            raise AppError(CodigoError.CONFIG_INVALIDA, "pending_run excede max_steps")

    def _validate_response_model(
        self,
        pending_run: PendingRun,
        response_model: type[BaseModel] | None,
    ) -> None:
        if pending_run.response_model_name != _response_model_name(response_model):
            raise AppError(
                CodigoError.CONFIG_INVALIDA,
                "response_model no coincide con el run pausado",
            )

    def _validate_pending_tools(self, tool_set: ToolSet, pending_run: PendingRun) -> None:
        for call in pending_run.calls:
            if tool_set.definition(call.name) is None:
                raise AppError(
                    CodigoError.CONFIG_INVALIDA,
                    f"falta tool para reanudar: {call.name}",
                )
        expected = {
            call.call_id for call in pending_run.calls if tool_set.needs_approval(call.name)
        }
        recorded = {item.call_id for item in pending_run.interruptions}
        if expected != recorded:
            raise AppError(
                CodigoError.CONFIG_INVALIDA,
                "las reglas de approval cambiaron desde la pausa",
            )

    def _validate_pending_record(self, record: SessionRecord, pending_run: PendingRun) -> None:
        session_id = pending_run.session_id
        if session_id is None or record.session_id != session_id:
            raise AppError(CodigoError.SESION_INVALIDA, "sesión de resume inválida")
        if record.provider != self._engine.provider or record.model != self._engine.model:
            raise AppError(
                CodigoError.SESION_ENGINE_DISTINTO,
                f"sesión ligada a {record.provider}/{record.model}",
            )
        active = record.active_turn
        expected_calls = tuple(call.call_id for call in pending_run.calls)
        if active is None or active.pending_call_ids != expected_calls or active.completed_call_ids:
            raise AppError(
                CodigoError.SESION_RECUPERACION_REQUERIDA,
                f"estado pausado inválido en sesión {record.session_id}",
            )
        if record.history != pending_run.history:
            raise AppError(
                CodigoError.SESION_CONFLICTO,
                f"la sesión {record.session_id} cambió durante la pausa",
            )

    async def _save_record(self, state: RunState) -> None:
        if state.record is None or state.lease_keeper is None:
            return
        state.record = await state.lease_keeper.save(state.record)

    async def _with_lease[T](
        self,
        state: RunState,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        if state.lease_keeper is None:
            return await operation()
        return await state.lease_keeper.run(operation)

    def _validate_record(self, record: SessionRecord, session_id: str) -> None:
        if record.session_id != session_id:
            raise AppError(
                CodigoError.SESION_INVALIDA,
                "el store devolvió una sesión con identidad distinta",
            )
        if record.active_turn is not None:
            raise AppError(
                CodigoError.SESION_RECUPERACION_REQUERIDA,
                f"turno incompleto en sesión {record.session_id}",
            )
        if record.provider != self._engine.provider or record.model != self._engine.model:
            raise AppError(
                CodigoError.SESION_ENGINE_DISTINTO,
                f"sesión ligada a {record.provider}/{record.model}",
            )


def _response_model_name(model: type[BaseModel] | None) -> str | None:
    if model is None:
        return None
    return f"{model.__module__}:{model.__qualname__}"
