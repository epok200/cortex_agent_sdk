from collections.abc import Iterable

import pytest
from pydantic import JsonValue

from cortex_agent_sdk import (
    CONTINUE,
    Agent,
    FinalOutput,
    PendingRun,
    fallback_answer,
    final_answer,
    tool,
)
from cortex_agent_sdk.engine import EngineRequest, EngineResult, ToolCall, Usage
from cortex_agent_sdk.history.models import TextPart, ToolCallPart, ToolResultPart, Turn
from cortex_agent_sdk.results import AgentExitReason
from cortex_agent_sdk.runtime import AgentOptions
from cortex_agent_sdk.sessions import MemorySessionStore


class ScriptedEngine:
    provider = "fake"
    model = "fake-model"

    def __init__(self, results: Iterable[EngineResult]) -> None:
        self.results = list(results)
        self.requests: list[EngineRequest] = []
        self.closed = False

    async def generate(self, request: EngineRequest) -> EngineResult:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("el engine fake se quedó sin respuestas")
        return self.results.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def _call(call_id: str, name: str, **arguments: JsonValue) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _result(
    *,
    text: str = "",
    calls: tuple[ToolCall, ...] = (),
    stop_reason: str | None = "completed",
) -> EngineResult:
    parts: list[TextPart | ToolCallPart] = []
    if text:
        parts.append(TextPart(text=text))
    parts.extend(
        ToolCallPart(call_id=call.call_id, name=call.name, arguments=call.arguments)
        for call in calls
    )
    return EngineResult(
        turn=Turn(role="assistant", parts=tuple(parts)),
        tool_calls=calls,
        usage=Usage(input_tokens=1, output_tokens=1),
        provider="fake",
        model="fake-model",
        stop_reason=stop_reason,
        raw={"text": text, "calls": len(calls)},
    )


def _tool_outputs(request: EngineRequest) -> list[ToolResultPart]:
    return [
        part
        for turn in request.history
        for part in turn.parts
        if isinstance(part, ToolResultPart)
    ]


@pytest.mark.asyncio
async def test_regular_tool_returns_to_model() -> None:
    executions: list[int] = []

    @tool
    async def double(value: int) -> str:
        executions.append(value)
        return str(value * 2)

    call = _call("call-1", "double", value=21)
    engine = ScriptedEngine((_result(calls=(call,)), _result(text="42")))

    async with Agent(engine, tools=(double,), own_engine=False) as agent:
        result = await agent.run("duplica 21")

    assert result.reason is AgentExitReason.COMPLETED
    assert result.text == "42"
    assert executions == [21]
    assert len(engine.requests) == 2


@pytest.mark.asyncio
async def test_final_answer_keeps_hard_stop() -> None:
    @final_answer
    async def answer(value: int) -> str:
        return str(value)

    call = _call("call-final", "answer", value=42)
    engine = ScriptedEngine((_result(calls=(call,)),))

    async with Agent(engine, tools=(answer,), own_engine=False) as agent:
        result = await agent.run("responde")

    assert result.reason is AgentExitReason.FINAL_ANSWER
    assert result.text == "42"
    assert len(engine.requests) == 1


@pytest.mark.asyncio
async def test_fallback_rescues_clean_empty_completion() -> None:
    @fallback_answer
    async def agenda() -> str:
        return "No tienes eventos mañana."

    call = _call("call-fallback", "agenda")
    engine = ScriptedEngine((_result(calls=(call,)), _result()))

    async with Agent(engine, tools=(agenda,), own_engine=False) as agent:
        result = await agent.run("qué tengo mañana")

    assert result.reason is AgentExitReason.FALLBACK_ANSWER
    assert result.text == "No tienes eventos mañana."
    assert len(engine.requests) == 2


@pytest.mark.asyncio
async def test_fallback_never_masks_incomplete_provider_stop() -> None:
    @fallback_answer
    async def agenda() -> str:
        return "fallback"

    call = _call("call-fallback", "agenda")
    engine = ScriptedEngine(
        (
            _result(calls=(call,)),
            _result(stop_reason="max_output_tokens"),
        )
    )

    async with Agent(engine, tools=(agenda,), own_engine=False) as agent:
        result = await agent.run("consulta")

    assert result.reason is AgentExitReason.COMPLETED
    assert result.text is None


@pytest.mark.asyncio
async def test_failed_fallback_is_never_a_final_candidate() -> None:
    @fallback_answer
    async def agenda(required: str) -> str:
        return required

    call = _call("call-invalid", "agenda")
    engine = ScriptedEngine((_result(calls=(call,)), _result()))

    async with Agent(engine, tools=(agenda,), own_engine=False) as agent:
        result = await agent.run("consulta")

    assert result.reason is AgentExitReason.COMPLETED
    assert result.text is None


@pytest.mark.asyncio
async def test_tool_result_policy_can_finish_dynamically() -> None:
    @tool
    async def lookup(value: int) -> str:
        return str(value)

    def policy(context) -> FinalOutput:
        assert context.outcomes[0].output == "7"
        return FinalOutput("policy-final")

    call = _call("call-policy", "lookup", value=7)
    engine = ScriptedEngine((_result(calls=(call,)),))

    async with Agent(
        engine,
        tools=(lookup,),
        own_engine=False,
        tool_result_policy=policy,
    ) as agent:
        result = await agent.run("consulta")

    assert result.reason is AgentExitReason.TOOL_POLICY
    assert result.text == "policy-final"
    assert len(engine.requests) == 1


@pytest.mark.asyncio
async def test_tool_result_policy_can_explicitly_continue() -> None:
    @tool
    async def lookup() -> str:
        return "dato"

    async def policy(_context):
        return CONTINUE

    call = _call("call-policy", "lookup")
    engine = ScriptedEngine((_result(calls=(call,)), _result(text="respuesta")))

    async with Agent(
        engine,
        tools=(lookup,),
        own_engine=False,
        tool_result_policy=policy,
    ) as agent:
        result = await agent.run("consulta")

    assert result.reason is AgentExitReason.COMPLETED
    assert result.text == "respuesta"
    assert len(engine.requests) == 2


@pytest.mark.asyncio
async def test_needs_approval_pauses_before_execution_and_resumes_after_approve() -> None:
    executions: list[str] = []

    @tool(needs_approval=True)
    async def delete_event(event_id: str) -> str:
        executions.append(event_id)
        return f"deleted:{event_id}"

    call = _call("call-delete", "delete_event", event_id="evt-1")
    engine = ScriptedEngine((_result(calls=(call,)), _result(text="Eliminado.")))

    async with Agent(engine, tools=(delete_event,), own_engine=False) as agent:
        paused = await agent.run("elimina el evento")

        assert paused.reason is AgentExitReason.APPROVAL_REQUIRED
        assert executions == []
        assert paused.pending_run is not None
        assert paused.interruptions[0].call_id == "call-delete"

        snapshot = PendingRun.from_json(paused.pending_run.to_json())
        snapshot.approve("call-delete")
        resumed = await agent.resume(snapshot)

    assert executions == ["evt-1"]
    assert resumed.reason is AgentExitReason.COMPLETED
    assert resumed.text == "Eliminado."
    assert len(engine.requests) == 2


@pytest.mark.asyncio
async def test_rejected_approval_never_executes_tool_and_is_visible_to_model() -> None:
    executions: list[str] = []

    @tool(needs_approval=True)
    async def delete_event(event_id: str) -> str:
        executions.append(event_id)
        return "deleted"

    call = _call("call-delete", "delete_event", event_id="evt-1")
    engine = ScriptedEngine((_result(calls=(call,)), _result(text="No lo eliminé.")))

    async with Agent(engine, tools=(delete_event,), own_engine=False) as agent:
        paused = await agent.run("elimina el evento")
        assert paused.pending_run is not None
        paused.pending_run.reject("call-delete", message="El usuario decidió conservar el evento.")
        resumed = await agent.resume(paused.pending_run)

    assert executions == []
    assert resumed.reason is AgentExitReason.COMPLETED
    outputs = _tool_outputs(engine.requests[1])
    assert len(outputs) == 1
    assert "conservar el evento" in outputs[0].output


@pytest.mark.asyncio
async def test_final_answer_can_require_approval_without_losing_hard_stop() -> None:
    executions: list[str] = []

    @final_answer(needs_approval=True)
    async def destructive(value: str) -> str:
        executions.append(value)
        return "confirmado"

    call = _call("call-final", "destructive", value="x")
    engine = ScriptedEngine((_result(calls=(call,)),))

    async with Agent(engine, tools=(destructive,), own_engine=False) as agent:
        paused = await agent.run("hazlo")
        assert paused.pending_run is not None
        paused.pending_run.approve("call-final")
        resumed = await agent.resume(paused.pending_run)

    assert executions == ["x"]
    assert resumed.reason is AgentExitReason.FINAL_ANSWER
    assert resumed.text == "confirmado"
    assert len(engine.requests) == 1


@pytest.mark.asyncio
async def test_session_keeps_active_turn_while_approval_is_pending() -> None:
    @tool(needs_approval=True)
    async def delete_event(event_id: str) -> str:
        return f"deleted:{event_id}"

    call = _call("call-session", "delete_event", event_id="evt-1")
    engine = ScriptedEngine((_result(calls=(call,)), _result(text="ok")))
    store = MemorySessionStore()

    async with Agent(
        engine,
        tools=(delete_event,),
        session_store=store,
        own_engine=False,
    ) as agent:
        paused = await agent.run("elimina", session_id="session-1")
        assert paused.pending_run is not None

        async with store.acquire("session-1", 1.0) as lease:
            record = await lease.load(lease.fencing_token)
        assert record is not None
        assert record.active_turn is not None
        assert record.active_turn.pending_call_ids == ("call-session",)

        paused.pending_run.approve("call-session")
        resumed = await agent.resume(paused.pending_run)

        async with store.acquire("session-1", 1.0) as lease:
            record = await lease.load(lease.fencing_token)

    await store.aclose()

    assert resumed.reason is AgentExitReason.COMPLETED
    assert record is not None
    assert record.active_turn is None


@pytest.mark.asyncio
async def test_fallback_is_not_used_when_max_steps_is_reached() -> None:
    @fallback_answer
    async def again() -> str:
        return "fallback"

    call_1 = _call("call-1", "again")
    call_2 = _call("call-2", "again")
    engine = ScriptedEngine((_result(calls=(call_1,)), _result(calls=(call_2,))))
    options = AgentOptions(max_steps=2)

    async with Agent(engine, tools=(again,), options=options, own_engine=False) as agent:
        result = await agent.run("sigue")

    assert result.reason is AgentExitReason.MAX_STEPS
    assert result.text is None
