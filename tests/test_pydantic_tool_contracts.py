import asyncio
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import requires, version
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

import cortex_agent_sdk.sessions.memory as memory_module
from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.sessions import MemorySessionStore, Session, SessionStore
from cortex_agent_sdk.sessions.codec import decode_messages, encode_messages


def _message(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)])


def test_import_raiz_no_carga_providers_ni_stores_opcionales() -> None:
    command = "\n".join(
        (
            "import sys",
            "import cortex_agent_sdk",
            "assert 'openai' not in sys.modules",
            "assert 'redis' not in sys.modules",
            "assert 'asyncpg' not in sys.modules",
            "assert 'pydantic_ai' not in sys.modules",
        )
    )

    subprocess.run([sys.executable, "-c", command], check=True)


def test_api_raiz_no_expone_runtime_pydantic() -> None:
    import cortex_agent_sdk

    assert not hasattr(cortex_agent_sdk, "Agent")
    assert MemorySessionStore
    assert Session
    assert SessionStore


def test_version_publica_coincide_con_metadata() -> None:
    import cortex_agent_sdk

    assert cortex_agent_sdk.__version__ == version("cortex-agent-sdk")


def test_dependencia_base_no_instala_providers() -> None:
    requirements = requires("cortex-agent-sdk") or []
    base = [
        item
        for item in requirements
        if item.startswith("pydantic-ai-slim") and ";" not in item
    ]

    assert base == ["pydantic-ai-slim<3,>=2.27"]


def test_session_replace_copia_el_historial() -> None:
    source: list[ModelMessage] = [_message("hola")]
    session = Session("s1", [])

    session.replace(source)
    source.clear()

    assert session.changed
    assert len(session.messages) == 1


def test_codec_usa_el_formato_nativo_de_pydantic_ai() -> None:
    messages: list[ModelMessage] = [
        _message("hola"),
        ModelResponse(
            parts=[ToolCallPart("consultar", {"id": 42}, "call-1")],
            model_name="test",
            provider_name="openai",
            provider_response_id="response-1",
            provider_details={"region": "test"},
            run_id="run-1",
            conversation_id="chat-1",
        ),
        ModelRequest(
            parts=[ToolReturnPart("consultar", "listo", "call-1")],
            run_id="run-1",
            conversation_id="chat-1",
        ),
    ]

    restored = decode_messages(encode_messages(messages))

    assert restored == messages


def test_codec_traduce_payload_invalido() -> None:
    with pytest.raises(AppError) as captured:
        decode_messages(b"no es json")

    assert captured.value.codigo is CodigoError.SESION_INVALIDA


async def test_memory_continua_un_agente_pydantic_sin_wrapper() -> None:
    history_sizes: list[int] = []

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        history_sizes.append(len(messages))
        return ModelResponse(parts=[TextPart(f"mensajes={len(messages)}")])

    agent = Agent(FunctionModel(respond))
    async with MemorySessionStore() as store:
        assert isinstance(store, SessionStore)

        async with store.turn("chat") as session:
            first = await agent.run("uno", message_history=session.messages)
            session.replace(first.all_messages())

        async with store.turn("chat") as session:
            second = await agent.run("dos", message_history=session.messages)
            session.replace(second.all_messages())

    assert first.output == "mensajes=1"
    assert second.output == "mensajes=3"
    assert history_sizes == [1, 3]


async def test_memory_no_guarda_sin_replace() -> None:
    store = MemorySessionStore()

    async with store.turn("chat") as session:
        assert not session.messages

    async with store.turn("chat") as session:
        assert not session.messages


async def test_memory_no_guarda_si_el_turno_falla() -> None:
    store = MemorySessionStore()

    with pytest.raises(RuntimeError):
        async with store.turn("chat") as session:
            session.replace([_message("no guardar")])
            raise RuntimeError("fallo de prueba")

    async with store.turn("chat") as session:
        assert not session.messages


async def test_memory_bloquea_la_misma_sesion() -> None:
    store = MemorySessionStore()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        async with store.turn("chat"):
            entered.set()
            await release.wait()

    task = asyncio.create_task(hold())
    await entered.wait()
    try:
        with pytest.raises(AppError) as captured:
            async with store.turn("chat", timeout_seconds=0.001):
                pass
        assert captured.value.codigo is CodigoError.SESION_OCUPADA

        async with store.turn("otro", timeout_seconds=0.01):
            pass
    finally:
        release.set()
        await task


async def test_memory_aplica_ttl_lru_y_reset() -> None:
    store = MemorySessionStore(max_sessions=2, ttl_seconds=0.02)
    for session_id in ("s1", "s2"):
        async with store.turn(session_id) as session:
            session.replace([_message(session_id)])

    async with store.turn("s1"):
        pass
    async with store.turn("s3") as session:
        session.replace([_message("s3")])
    async with store.turn("s2") as session:
        assert not session.messages

    await store.reset("s1")
    async with store.turn("s1") as session:
        assert not session.messages

    await asyncio.sleep(0.03)
    async with store.turn("s3") as session:
        assert not session.messages


async def test_memory_lru_no_expulsa_turno_activo() -> None:
    store = MemorySessionStore(max_sessions=1)

    with pytest.raises(RuntimeError, match="caída simulada"):
        async with store.turn("efecto") as active_session:
            await active_session.start_tool_turn(
                [ToolCallPart("crear", {}, "call-1")]
            )
            async with store.turn("lectura") as other_session:
                other_session.replace([_message("otra sesión")])
            raise RuntimeError("caída simulada")

    with pytest.raises(AppError) as captured:
        async with store.turn("efecto"):
            pass

    assert captured.value.codigo is CodigoError.SESION_RECUPERACION_REQUERIDA


async def test_memory_lru_no_expulsa_sesion_adquirida() -> None:
    store = MemorySessionStore(max_sessions=1)
    history = _message("historial de A")

    async with store.turn("sesion-a") as session:
        session.replace([history])

    async with store.turn("sesion-a"), store.turn("sesion-b") as session_b:
        session_b.replace([_message("historial de B")])

    async with store.turn("sesion-a") as session:
        assert session.messages == [history]


async def test_memory_lru_no_autoexpulsa_checkpoint_actual() -> None:
    store = MemorySessionStore(max_sessions=1)
    checkpoint = ModelRequest(
        parts=[ToolReturnPart("crear", "efecto confirmado", "call-b")]
    )

    with pytest.raises(RuntimeError, match="caída posterior"):
        async with store.turn("sesion-a") as session_a:
            await session_a.start_tool_turn(
                [ToolCallPart("crear", {}, "call-a")]
            )
            async with store.turn("sesion-b") as session_b:
                await session_b.start_tool_turn(
                    [ToolCallPart("crear", {}, "call-b")]
                )
                await session_b.checkpoint([checkpoint])
                raise RuntimeError("caída posterior")

    async with store.turn("sesion-b") as session_b:
        assert session_b.messages == [checkpoint]

    with pytest.raises(AppError) as captured:
        async with store.turn("sesion-a"):
            pass

    assert captured.value.codigo is CodigoError.SESION_RECUPERACION_REQUERIDA


async def test_memory_lru_purga_marcadores_expirados(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(memory_module, "time", fake_time)
    store = MemorySessionStore(max_sessions=1, ttl_seconds=1)

    for index in range(5):
        async with store.turn(f"sesion-{index}") as session:
            await session.start_tool_turn(
                [ToolCallPart("crear", {}, f"call-{index}")]
            )
        clock[0] += 2

    assert list(store._entries) == ["sesion-4"]


async def test_memory_mantiene_marcador_durante_tool_larga(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(memory_module, "time", fake_time)
    store = MemorySessionStore(max_sessions=1, ttl_seconds=1)

    with pytest.raises(RuntimeError, match="caída tardía"):
        async with store.turn("chat") as session:
            await session.start_tool_turn(
                [ToolCallPart("crear", {}, "call-1")]
            )
            clock[0] = 5
            async with store.turn("otra") as other:
                other.replace([_message("otra sesión")])
            raise RuntimeError("caída tardía")

    with pytest.raises(AppError) as captured:
        async with store.turn("chat"):
            pass
    assert captured.value.codigo is CodigoError.SESION_RECUPERACION_REQUERIDA

    clock[0] = 7
    async with store.turn("chat") as session:
        assert not session.messages


async def test_memory_session_no_persiste_fuera_de_su_turno() -> None:
    store = MemorySessionStore()
    current = _message("historial actual")
    stale = _message("historial obsoleto")

    async with store.turn("chat") as old_session:
        captured_checkpoint = old_session._save_checkpoint

    assert captured_checkpoint is not None
    async with store.turn("chat") as new_session:
        new_session.replace([current])

    with pytest.raises(AppError) as checkpoint_error:
        await old_session.checkpoint([stale])
    with pytest.raises(AppError) as tool_error:
        await old_session.start_tool_turn([ToolCallPart("crear", {}, "call-old")])
    with pytest.raises(AppError) as owner_error:
        await captured_checkpoint([stale])

    assert checkpoint_error.value.codigo is CodigoError.CONFIG_INVALIDA
    assert tool_error.value.codigo is CodigoError.CONFIG_INVALIDA
    assert owner_error.value.codigo is CodigoError.SESION_INVALIDA
    async with store.turn("chat") as session:
        assert session.messages == [current]


async def test_memory_no_cierra_con_turnos_activos() -> None:
    store = MemorySessionStore()

    async with store.turn("chat"):
        with pytest.raises(AppError) as captured:
            await store.aclose()

    assert captured.value.codigo is CodigoError.SESION_OCUPADA
    await store.aclose()


@dataclass(slots=True)
class _Deps:
    factor: int


async def test_pydantic_conserva_tipado_tools_y_deps_privadas() -> None:
    schemas: list[dict] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        schemas.append(info.function_tools[0].parameters_json_schema)
        parts = (part for message in messages for part in message.parts)
        has_result = any(isinstance(part, ToolReturnPart) for part in parts)
        if has_result:
            return ModelResponse(parts=[TextPart("resultado listo")])
        return ModelResponse(parts=[ToolCallPart("multiplicar", {"valor": 4}, "call-1")])

    agent = Agent(FunctionModel(respond), deps_type=_Deps)

    @agent.tool
    async def multiplicar(context: RunContext[_Deps], valor: int) -> int:
        return valor * context.deps.factor

    async with MemorySessionStore() as store:
        async with store.turn("chat") as session:
            result = await agent.run(
                "Multiplica cuatro.",
                deps=_Deps(factor=3),
                message_history=session.messages,
            )
            session.replace(result.all_messages())

        async with store.turn("chat") as session:
            restored = session.messages

    restored_parts = (part for message in restored for part in message.parts)

    assert result.output == "resultado listo"
    assert any(isinstance(part, ToolReturnPart) for part in restored_parts)
    assert schemas[0]["properties"] == {"valor": {"type": "integer"}}
    assert "context" not in schemas[0]["properties"]
