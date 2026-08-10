import pytest
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolFailed,
)
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from cortex_agent_sdk.capabilities import last_tool_result_fallback


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
