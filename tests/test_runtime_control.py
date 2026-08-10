import asyncio
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import cast
from uuid import uuid4

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from redis.asyncio import Redis

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.redis import scripts
from cortex_agent_sdk.redis.store import RedisSessionStore
from cortex_agent_sdk.sessions import MemorySessionStore


@dataclass(slots=True)
class _Value:
    content: str | bytes
    expires_at: float


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, _Value] = {}
        self.reject_renewals = False
        self.reject_releases = False
        self.closed = False

    async def eval(self, script: str, key_count: int, *values: object) -> object:
        keys = cast(tuple[str, ...], values[:key_count])
        arguments = values[key_count:]

        if script == scripts.ACQUIRE:
            return self._acquire(keys[0], cast(str, arguments[0]), cast(int, arguments[1]))
        if script == scripts.RENEW:
            return self._renew(keys[0], cast(str, arguments[0]), cast(int, arguments[1]))
        if script == scripts.LOAD:
            return self._load(keys, cast(str, arguments[0]), cast(int, arguments[1]))
        if script == scripts.SAVE:
            return self._save(
                keys,
                cast(str, arguments[0]),
                cast(bytes, arguments[1]),
                cast(int, arguments[2]),
            )
        if script == scripts.RESET:
            return self._reset(keys, cast(str, arguments[0]))
        if script == scripts.RELEASE:
            return self._release(keys[0], cast(str, arguments[0]))
        raise AssertionError("script desconocido")

    async def aclose(self) -> None:
        self.closed = True

    def corrupt_history(self, prefix: str, session_id: str) -> None:
        digest = sha256(session_id.encode()).hexdigest()
        key = f"{prefix}:{{{digest}}}:history"
        self._set(key, "payload inválido".encode(), 1_000)

    def _acquire(self, key: str, owner: str, ttl: int) -> int:
        if self._get(key) is not None:
            return 0
        self._set(key, owner, ttl)
        return 1

    def _renew(self, key: str, owner: str, ttl: int) -> int:
        if self.reject_renewals or self._get(key) != owner:
            return 0
        self._set(key, owner, ttl)
        return 1

    def _load(self, keys: tuple[str, ...], owner: str, ttl: int) -> list[object]:
        if self._get(keys[0]) != owner:
            return [0]
        payload = self._get(keys[1])
        if payload is None:
            return [1]
        self._set(keys[1], payload, ttl)
        return [1, payload]

    def _save(self, keys: tuple[str, ...], owner: str, payload: bytes, ttl: int) -> int:
        if self._get(keys[0]) != owner:
            return 0
        self._set(keys[1], payload, ttl)
        return 1

    def _reset(self, keys: tuple[str, ...], owner: str) -> int:
        if self._get(keys[0]) != owner:
            return 0
        self.values.pop(keys[1], None)
        return 1

    def _release(self, key: str, owner: str) -> int:
        if self.reject_releases or self._get(key) != owner:
            return 0
        self.values.pop(key, None)
        return 1

    def _get(self, key: str) -> str | bytes | None:
        value = self.values.get(key)
        if value is None:
            return None
        if value.expires_at <= time.monotonic():
            del self.values[key]
            return None
        return value.content

    def _set(self, key: str, content: str | bytes, ttl_milliseconds: int) -> None:
        expires_at = time.monotonic() + ttl_milliseconds / 1_000
        self.values[key] = _Value(content, expires_at)


class _HangingRedis:
    async def eval(self, *_: object) -> object:
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        return None


class _SlowBusyRedis:
    async def eval(self, *_: object) -> object:
        await asyncio.sleep(0.03)
        return 0

    async def aclose(self) -> None:
        return None


class _SlowReleaseRedis(_FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.release_started = asyncio.Event()
        self.finish_release = asyncio.Event()

    async def eval(self, script: str, key_count: int, *values: object) -> object:
        if script == scripts.RELEASE:
            self.release_started.set()
            await self.finish_release.wait()
        return await super().eval(script, key_count, *values)


class _SlowRenewalCleanupStore(RedisSessionStore):
    def __init__(self, client: _FakeRedis) -> None:
        super().__init__(client=cast(Redis, client))
        self.cleanup_started = asyncio.Event()
        self.finish_cleanup = asyncio.Event()

    async def _renew_loop(
        self,
        session_id: str,
        owner: str,
        owner_task: asyncio.Task,
    ) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            self.cleanup_started.set()
            await self.finish_cleanup.wait()


class _DoubleCancelStore(RedisSessionStore):
    async def _renew_loop(
        self,
        session_id: str,
        owner: str,
        owner_task: asyncio.Task,
    ) -> None:
        await asyncio.sleep(0.01)
        owner_task.cancel()
        owner_task.cancel()
        raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")


def _store(
    client: _FakeRedis,
    *,
    prefix: str = "test:sessions",
    lease_seconds: float = 0.03,
) -> RedisSessionStore:
    return RedisSessionStore(
        client=cast(Redis, client),
        key_prefix=prefix,
        lease_seconds=lease_seconds,
        retry_interval_seconds=0.002,
        io_timeout_seconds=0.1,
    )


async def test_redis_persiste_resetea_y_no_cierra_cliente_inyectado() -> None:
    client = _FakeRedis()
    first = _store(client)
    second = _store(client)
    message = ModelRequest(parts=[UserPromptPart("hola")])

    async with first.turn("chat") as session:
        session.replace([message])
    async with second.turn("chat") as session:
        assert session.messages == [message]

    await second.reset("chat")
    async with first.turn("chat") as session:
        assert not session.messages

    await first.aclose()
    await second.aclose()
    assert not client.closed


async def test_redis_renueva_el_lease_durante_el_turno() -> None:
    client = _FakeRedis()
    first = _store(client)
    second = _store(client)

    async with first.turn("chat") as session:
        await asyncio.sleep(0.08)
        with pytest.raises(AppError) as captured:
            async with second.turn("chat", timeout_seconds=0.01):
                pass
        session.replace([ModelRequest(parts=[UserPromptPart("guardado")])])

    assert captured.value.codigo is CodigoError.SESION_OCUPADA
    async with second.turn("chat") as session:
        assert len(session.messages) == 1


async def test_redis_no_guarda_si_pierde_el_lease() -> None:
    client = _FakeRedis()
    store = _store(client)

    with pytest.raises(AppError) as captured:
        async with store.turn("chat") as session:
            session.replace([ModelRequest(parts=[UserPromptPart("no guardar")])])
            client.reject_renewals = True
            await asyncio.sleep(0.02)

    assert captured.value.codigo is CodigoError.SESION_LEASE_PERDIDO
    client.reject_renewals = False
    async with store.turn("chat") as session:
        assert not session.messages


async def test_redis_interrumpe_el_turno_si_falla_la_renovacion() -> None:
    client = _FakeRedis()
    store = _store(client)
    entered = asyncio.Event()

    async def run() -> None:
        with pytest.raises(AppError) as captured:
            async with store.turn("chat"):
                client.reject_renewals = True
                entered.set()
                await asyncio.Event().wait()
        assert captured.value.codigo is CodigoError.SESION_LEASE_PERDIDO

    task = asyncio.create_task(run())
    await entered.wait()
    await asyncio.wait_for(task, timeout=0.1)


async def test_redis_reporta_si_no_puede_liberar_el_lease() -> None:
    client = _FakeRedis()
    store = _store(client)

    with pytest.raises(AppError) as captured:
        async with store.turn("chat"):
            client.reject_releases = True

    assert captured.value.codigo is CodigoError.SESION_LEASE_PERDIDO


async def test_redis_no_oculta_el_error_del_turno_si_release_tambien_falla() -> None:
    client = _FakeRedis()
    store = _store(client)

    with pytest.raises(RuntimeError, match="fallo principal"):
        async with store.turn("chat"):
            client.reject_releases = True
            raise RuntimeError("fallo principal")


async def test_redis_preserva_cancelacion_externa_durante_cleanup() -> None:
    store = _SlowRenewalCleanupStore(_FakeRedis())

    async def run() -> None:
        async with store.turn("chat"):
            pass

    task = asyncio.create_task(run())
    await store.cleanup_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    store.finish_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await store.aclose()


async def test_redis_no_filtra_turno_si_cancelan_durante_release() -> None:
    client = _SlowReleaseRedis()
    store = _store(client)

    async def run() -> None:
        async with store.turn("chat"):
            pass

    task = asyncio.create_task(run())
    await client.release_started.wait()
    task.cancel()
    client.finish_release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    await store.aclose()


async def test_redis_cancelacion_externa_gana_a_falla_de_heartbeat() -> None:
    store = _DoubleCancelStore(client=cast(Redis, _FakeRedis()))

    async def run() -> None:
        async with store.turn("chat"):
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    await store.aclose()


async def test_redis_traduce_historial_corrupto() -> None:
    client = _FakeRedis()
    prefix = "test:corrupt"
    client.corrupt_history(prefix, "chat")
    store = _store(client, prefix=prefix)

    with pytest.raises(AppError) as captured:
        async with store.turn("chat"):
            pass

    assert captured.value.codigo is CodigoError.SESION_INVALIDA


async def test_redis_limita_cada_operacion_de_io() -> None:
    store = RedisSessionStore(
        client=cast(Redis, _HangingRedis()),
        io_timeout_seconds=0.01,
    )

    with pytest.raises(AppError) as captured:
        async with store.turn("chat", timeout_seconds=1):
            pass

    assert captured.value.codigo is CodigoError.SESION_STORE_FALLO


async def test_redis_respeta_el_timeout_total_de_adquisicion() -> None:
    store = RedisSessionStore(
        client=cast(Redis, _SlowBusyRedis()),
        retry_interval_seconds=0.05,
        io_timeout_seconds=0.1,
    )
    started_at = time.monotonic()

    with pytest.raises(AppError) as captured:
        async with store.turn("chat", timeout_seconds=0.05):
            pass

    elapsed = time.monotonic() - started_at
    assert captured.value.codigo is CodigoError.SESION_OCUPADA
    assert elapsed < 0.075


async def test_redis_real_persiste_y_excluye_entre_procesos() -> None:
    url = os.getenv("CORTEX_TEST_REDIS_URL")
    if url is None:
        pytest.skip("requiere CORTEX_TEST_REDIS_URL")

    prefix = f"cortex:test:{uuid4().hex}"
    first = RedisSessionStore(url, key_prefix=prefix)
    second = RedisSessionStore(url, key_prefix=prefix)
    session_id = "chat"

    try:
        await first.reset(session_id)
        async with first.turn(session_id) as session:
            session.replace([ModelRequest(parts=[UserPromptPart("hola")])])
            with pytest.raises(AppError) as captured:
                async with second.turn(session_id, timeout_seconds=0.01):
                    pass
            assert captured.value.codigo is CodigoError.SESION_OCUPADA

        async with second.turn(session_id) as session:
            assert len(session.messages) == 1

        await second.reset(session_id)
        async with first.turn(session_id) as session:
            assert not session.messages
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.smoke
async def test_openai_responses_real() -> None:
    if os.getenv("CORTEX_RUN_OPENAI_SMOKE") != "1":
        pytest.skip("requiere CORTEX_RUN_OPENAI_SMOKE=1")

    agent = Agent("openai-responses:gpt-5.6-luna")
    sessions = MemorySessionStore()
    async with agent, sessions:
        async with sessions.turn("smoke") as session:
            first = await agent.run(
                "Recuerda exactamente esta clave: azul7. Responde sólo: ok",
                message_history=session.messages,
            )
            session.replace(first.all_messages())

        async with sessions.turn("smoke") as session:
            result = await agent.run(
                "Responde sólo con la clave que te pedí recordar.",
                message_history=session.messages,
            )
            session.replace(result.all_messages())

    assert "azul7" in result.output.strip().lower()
