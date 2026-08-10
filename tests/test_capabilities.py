import asyncio

import pytest
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RunContext,
    TextPart,
    ToolFailed,
)
from pydantic_ai.capabilities import WrapModelRequestHandler
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from cortex_agent_sdk.capabilities import last_tool_result_fallback, session_checkpoints
from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.sessions import MemorySessionStore, Session


class _ConfirmedOutput(BaseModel):
    status: str


def test_session_checkpoints_se_construye_en_python_313() -> None:
    hooks = session_checkpoints(Session("chat", []), effect_tools="crear")

    assert hooks.after_node_run is not None


async def test_fallback_usa_ultimo_resultado_exitoso_del_run() -> None:
    requests = 0

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        parts = (part for message in messages for part in message.parts)
        if not any(isinstance(part, ToolReturnPart) for part in parts):
            return ModelResponse(parts=[ToolCallPart("responder", {}, "call-1")])
        return ModelResponse(parts=[], finish_reason="stop")

    agent = Agent(
        FunctionModel(respond),
        capabilities=[last_tool_result_fallback({"responder"})],
    )

    @agent.tool_plain
    def responder() -> str:
        return "resultado listo"

    result = await agent.run("Responde con la herramienta.")

    assert result.output == "resultado listo"
    assert requests == 2


async def test_fallback_no_reutiliza_resultado_de_otro_run() -> None:
    responses = iter(
        (
            ModelResponse(parts=[ToolCallPart("responder", {}, "call-1")]),
            ModelResponse(parts=[], finish_reason="stop"),
            ModelResponse(parts=[], finish_reason="stop"),
            ModelResponse(parts=[TextPart("respuesta nueva")]),
        )
    )

    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return next(responses)

    agent = Agent(
        FunctionModel(respond),
        capabilities=[last_tool_result_fallback({"responder"})],
    )

    @agent.tool_plain
    def responder() -> str:
        return "resultado anterior"

    first = await agent.run("Primer turno.")
    second = await agent.run("Segundo turno.", message_history=first.all_messages())

    assert first.output == "resultado anterior"
    assert second.output == "respuesta nueva"


async def test_fallback_ignora_resultados_fallidos() -> None:
    responses = iter(
        (
            ModelResponse(parts=[ToolCallPart("fallar", {}, "call-1")]),
            ModelResponse(parts=[], finish_reason="stop"),
            ModelResponse(parts=[TextPart("recuperado por el modelo")]),
        )
    )

    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return next(responses)

    agent = Agent(
        FunctionModel(respond),
        capabilities=[last_tool_result_fallback({"fallar"})],
    )

    @agent.tool_plain
    def fallar() -> str:
        raise ToolFailed("fallo esperado")

    result = await agent.run("Ejecuta la herramienta.")

    assert result.output == "recuperado por el modelo"


async def test_fallback_recupera_si_provider_falla_despues_de_tool_exitosa() -> None:
    failure = ModelAPIError("test-model", "provider caído")

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        parts = (part for message in messages for part in message.parts)
        if any(isinstance(part, ToolReturnPart) for part in parts):
            raise failure
        return ModelResponse(parts=[ToolCallPart("responder", {}, "call-1")])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[last_tool_result_fallback("responder")],
    )

    @agent.tool_plain
    def responder() -> str:
        return "resultado verificado"

    result = await agent.run("Responde con la herramienta.")

    assert result.output == "resultado verificado"


async def test_fallback_propaga_error_si_no_hay_tool_elegible() -> None:
    failure = ModelAPIError("test-model", "provider caído")

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        parts = (part for message in messages for part in message.parts)
        if any(isinstance(part, ToolReturnPart) for part in parts):
            raise failure
        return ModelResponse(parts=[ToolCallPart("consultar", {}, "call-1")])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[last_tool_result_fallback("responder")],
    )

    @agent.tool_plain
    def consultar() -> str:
        return "resultado no elegible"

    with pytest.raises(ModelAPIError) as captured:
        await agent.run("Consulta con la herramienta.")

    assert captured.value is failure


async def test_fallback_no_atrapa_error_local_despues_de_tool_exitosa() -> None:
    failure = RuntimeError("fallo de hook local")

    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("responder", {}, "call-1")])

    async def fail_after_tool(
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        parts = (part for message in request_context.messages for part in message.parts)
        if any(isinstance(part, ToolReturnPart) for part in parts):
            raise failure
        return await handler(request_context)

    agent = Agent(
        FunctionModel(respond),
        capabilities=[
            last_tool_result_fallback("responder"),
            Hooks(model_request=fail_after_tool),
        ],
    )

    @agent.tool_plain
    def responder() -> str:
        return "resultado verificado"

    with pytest.raises(RuntimeError) as captured:
        await agent.run("Responde con la herramienta.")

    assert captured.value is failure


async def test_fallback_no_atrapa_error_de_tool() -> None:
    failure = RuntimeError("tool falló")

    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("crear", {}, "call-1")])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[last_tool_result_fallback("crear")],
    )

    @agent.tool_plain
    def crear() -> str:
        raise failure

    with pytest.raises(RuntimeError) as captured:
        await agent.run("Crea el recurso.")

    assert captured.value is failure


async def test_checkpoint_conserva_tool_return_si_falla_el_provider() -> None:
    failure = RuntimeError("provider caído")

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        parts = (part for message in messages for part in message.parts)
        if any(isinstance(part, ToolReturnPart) for part in parts):
            raise failure
        return ModelResponse(parts=[ToolCallPart("crear", {}, "call-1")])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def crear() -> str:
        return "efecto confirmado"

    store = MemorySessionStore()
    async with store.turn("chat") as session:
        with pytest.raises(RuntimeError, match="provider caído"):
            await agent.run(
                "Crea el recurso.",
                message_history=session.messages,
                capabilities=[session_checkpoints(session, effect_tools="crear")],
            )

    async with store.turn("chat") as session:
        returns = (
            part
            for message in session.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        )
        assert [part.content for part in returns] == ["efecto confirmado"]


async def test_checkpoint_bloquea_turno_ambiguo_si_tool_interrumpe() -> None:
    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("crear", {}, "call-1")])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def crear() -> str:
        raise RuntimeError("caída después del efecto")

    store = MemorySessionStore()
    with pytest.raises(RuntimeError, match="caída después del efecto"):
        async with store.turn("chat") as session:
            await agent.run(
                "Crea el recurso.",
                message_history=session.messages,
                capabilities=[session_checkpoints(session, effect_tools="crear")],
            )

    with pytest.raises(AppError) as captured:
        async with store.turn("chat"):
            pass

    assert captured.value.codigo is CodigoError.SESION_RECUPERACION_REQUERIDA


async def test_checkpoint_no_marca_como_efecto_una_tool_de_lectura() -> None:
    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("consultar", {}, "call-1")])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def consultar() -> str:
        raise RuntimeError("lectura interrumpida")

    store = MemorySessionStore()
    with pytest.raises(RuntimeError, match="lectura interrumpida"):
        async with store.turn("chat") as session:
            await agent.run(
                "Consulta el recurso.",
                message_history=session.messages,
                capabilities=[session_checkpoints(session, effect_tools="crear")],
            )

    async with store.turn("chat") as session:
        assert not session.messages


async def test_checkpoint_no_guarda_turno_sin_tools() -> None:
    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("respuesta final")])

    agent = Agent(FunctionModel(respond))
    store = MemorySessionStore()
    async with store.turn("chat") as session:
        result = await agent.run(
            "Responde sin tools.",
            message_history=session.messages,
            capabilities=[session_checkpoints(session, effect_tools="crear")],
        )

    assert result.output == "respuesta final"
    async with store.turn("chat") as session:
        assert not session.messages


async def test_checkpoint_conserva_marcador_si_cancelan_tool_de_efecto() -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()

    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("crear", {}, "call-1")])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    async def crear() -> str:
        started.set()
        await blocked.wait()
        return "efecto confirmado"

    store = MemorySessionStore()

    async def run_effect() -> None:
        async with store.turn("chat") as session:
            await agent.run(
                "Crea el recurso.",
                message_history=session.messages,
                capabilities=[session_checkpoints(session, effect_tools="crear")],
            )

    task = asyncio.create_task(run_effect())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(AppError) as captured:
        async with store.turn("chat"):
            pass

    assert captured.value.codigo is CodigoError.SESION_RECUPERACION_REQUERIDA


async def test_checkpoint_guarda_tools_de_efecto_y_output_en_end() -> None:
    def respond(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart("crear", {}, "call-effect"),
                ToolCallPart(
                    info.output_tools[0].name,
                    {"status": "listo"},
                    "call-output",
                ),
            ]
        )

    agent = Agent(FunctionModel(respond), output_type=_ConfirmedOutput)

    @agent.tool_plain
    def crear() -> str:
        return "efecto confirmado"

    store = MemorySessionStore()
    async with store.turn("chat") as session:
        result = await agent.run(
            "Crea el recurso.",
            message_history=session.messages,
            capabilities=[session_checkpoints(session, effect_tools="crear")],
        )

    async with store.turn("chat") as session:
        returns = [
            part
            for message in session.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]

    assert result.output == _ConfirmedOutput(status="listo")
    assert [(part.tool_name, part.content) for part in returns] == [
        ("crear", "efecto confirmado"),
        ("final_result", "Final result processed."),
    ]


@pytest.mark.parametrize(
    "tool_error",
    (ToolFailed("efecto fallido"), ModelRetry("reintenta el efecto")),
    ids=("tool-failed", "model-retry"),
)
async def test_checkpoint_conserva_efecto_fallido_tras_lectura(
    tool_error: Exception,
) -> None:
    responses = [
        ModelResponse(parts=[ToolCallPart("crear", {}, "call-effect")]),
        ModelResponse(parts=[ToolCallPart("consultar", {}, "call-read")]),
    ]
    provider_failure = ModelAPIError("test-model", "provider caído")
    effects = 0

    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if responses:
            return responses.pop(0)
        raise provider_failure

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def crear() -> str:
        nonlocal effects
        effects += 1
        raise tool_error

    @agent.tool_plain
    def consultar() -> str:
        return "lectura lista"

    store = MemorySessionStore()
    with pytest.raises(ModelAPIError) as captured:
        async with store.turn("chat") as session:
            await agent.run(
                "Crea y consulta.",
                message_history=session.messages,
                capabilities=[
                    session_checkpoints(session, effect_tools="crear"),
                    last_tool_result_fallback("crear"),
                ],
            )

    assert effects == 1
    assert captured.value is provider_failure
    with pytest.raises(AppError) as recovery:
        async with store.turn("chat"):
            pass
    assert recovery.value.codigo is CodigoError.SESION_RECUPERACION_REQUERIDA


@pytest.mark.parametrize(
    "tool_error",
    (ToolFailed("segunda tool falló"), ModelRetry("reintenta segunda tool")),
    ids=("tool-failed", "model-retry"),
)
async def test_fallback_no_recupera_run_con_resultado_parcial(
    tool_error: Exception,
) -> None:
    responses = [
        ModelResponse(parts=[ToolCallPart("primera", {}, "call-1")]),
        ModelResponse(parts=[ToolCallPart("segunda", {}, "call-2")]),
    ]
    provider_failure = ModelAPIError("test-model", "provider caído")

    def respond(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if responses:
            return responses.pop(0)
        raise provider_failure

    agent = Agent(
        FunctionModel(respond),
        capabilities=[last_tool_result_fallback("primera")],
    )

    @agent.tool_plain
    def primera() -> str:
        return "primer resultado"

    @agent.tool_plain
    def segunda() -> str:
        raise tool_error

    with pytest.raises(ModelAPIError) as captured:
        await agent.run("Ejecuta ambas tools.")

    assert captured.value is provider_failure


@pytest.mark.parametrize(
    "response",
    (
        ModelResponse(parts=[TextPart("respuesta del modelo")], finish_reason="stop"),
        ModelResponse(parts=[ToolCallPart("continuar", {}, "call-2")]),
        ModelResponse(parts=[], finish_reason="length"),
    ),
)
async def test_fallback_respeta_respuesta_del_modelo(response: ModelResponse) -> None:
    returned = await _apply_fallback(
        ToolReturnPart("responder", "resultado listo", "call-1"),
        response,
    )

    assert returned == response


@pytest.mark.parametrize(
    "part",
    (
        ToolReturnPart("otra_tool", "resultado listo", "call-1"),
        ToolReturnPart("responder", "   ", "call-1"),
        ToolReturnPart("responder", {"resultado": "listo"}, "call-1"),
        ToolReturnPart("responder", "fallo", "call-1", outcome="failed"),
    ),
)
async def test_fallback_ignora_resultado_no_elegible(part: ToolReturnPart) -> None:
    response = ModelResponse(parts=[], finish_reason="stop")

    returned = await _apply_fallback(part, response)

    assert returned == response


async def _apply_fallback(
    part: ToolReturnPart,
    response: ModelResponse,
) -> ModelResponse:
    model = TestModel()
    run_id = "run-actual"
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), run_id=run_id)
    request_context = ModelRequestContext(
        model=model,
        messages=[ModelRequest(parts=[part], run_id=run_id)],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    hooks = last_tool_result_fallback("responder")
    return await hooks.after_model_request(
        ctx,
        request_context=request_context,
        response=response,
    )
