import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import Self, cast
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.redis import scripts
from cortex_agent_sdk.sessions.codec import decode_messages, encode_messages
from cortex_agent_sdk.sessions.models import Session

type _RedisArgument = str | bytes | int | float


@dataclass(frozen=True, slots=True)
class _RedisKeys:
    lease: str
    history: str


class RedisSessionStore:
    """Sesiones distribuidas con lease renovable y guardado atómico."""

    def __init__(
        self,
        url: str | None = None,
        *,
        client: Redis | None = None,
        key_prefix: str = "cortex:pydantic:sessions",
        ttl_seconds: float = 86_400,
        lease_seconds: float = 30,
        retry_interval_seconds: float = 0.05,
        io_timeout_seconds: float = 10,
    ) -> None:
        if (url is None) == (client is None):
            raise AppError(CodigoError.CONFIG_INVALIDA, "indica url o client de Redis")
        durations = (ttl_seconds, lease_seconds, retry_interval_seconds, io_timeout_seconds)
        if not key_prefix or min(durations) <= 0:
            raise AppError(CodigoError.CONFIG_INVALIDA, "configuración de Redis inválida")

        self._client = client or Redis.from_url(cast(str, url), decode_responses=False)
        self._owns_client = client is None
        self._key_prefix = key_prefix.rstrip(":")
        self._ttl_milliseconds = _milliseconds(ttl_seconds)
        self._lease_milliseconds = _milliseconds(lease_seconds)
        self._retry_interval_seconds = retry_interval_seconds
        self._io_timeout_seconds = io_timeout_seconds
        self._guard = asyncio.Lock()
        self._active_turns = 0
        self._closed = False

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    @asynccontextmanager
    async def turn(
        self,
        session_id: str,
        timeout_seconds: float = 5.0,
    ) -> AsyncIterator[Session]:
        async with self._lease(session_id, timeout_seconds) as lease:
            payload = await self._load(session_id, lease.owner)
            messages = decode_messages(payload) if payload is not None else []
            session = Session(session_id, messages)
            yield session
            self._raise_renewal_failure(lease)
            if session.changed:
                await self._save(session_id, lease.owner, encode_messages(session.messages))

    async def reset(self, session_id: str, timeout_seconds: float = 5.0) -> None:
        async with self._lease(session_id, timeout_seconds) as lease:
            keys = self._keys(session_id)
            result = await self._eval(scripts.RESET, (keys.lease, keys.history), lease.owner)
            if int(cast(int, result)) != 1:
                raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def aclose(self) -> None:
        async with self._guard:
            if self._closed:
                return
            if self._active_turns:
                raise AppError(CodigoError.SESION_OCUPADA, "hay turnos de sesión activos")
            if self._owns_client:
                await self._close_client()
            self._closed = True

    @asynccontextmanager
    async def _lease(
        self,
        session_id: str,
        timeout_seconds: float,
    ) -> AsyncIterator["_Lease"]:
        self._validate_turn(session_id, timeout_seconds)
        await self._register_turn()
        owner = uuid4().hex
        acquired = False
        renewal: asyncio.Task[None] | None = None
        body_error: BaseException | None = None
        try:
            await self._acquire(session_id, owner, timeout_seconds)
            acquired = True
            owner_task = asyncio.current_task()
            if owner_task is None:
                raise RuntimeError("turno sin task de asyncio")
            renewal = asyncio.create_task(self._renew_loop(session_id, owner, owner_task))
            try:
                yield _Lease(owner, renewal, owner_task)
            except asyncio.CancelledError as error:
                renewal_failure = self._renewal_failure(renewal)
                if renewal_failure is not None and owner_task.cancelling() == 1:
                    owner_task.uncancel()
                    body_error = renewal_failure
                    raise renewal_failure from error
                body_error = error
                raise
            except BaseException as error:
                body_error = error
                raise
        finally:
            cleanup = asyncio.create_task(self._cleanup_lease(session_id, owner, renewal, acquired))
            cleanup_error = await self._wait_for_cleanup(cleanup)
            if body_error is None and cleanup_error is not None:
                raise cleanup_error

    async def _cleanup_lease(
        self,
        session_id: str,
        owner: str,
        renewal: asyncio.Task[None] | None,
        acquired: bool,
    ) -> AppError | None:
        renewal_error: AppError | None = None
        release_error: AppError | None = None
        try:
            if renewal is not None:
                renewal.cancel()
                result = await asyncio.gather(renewal, return_exceptions=True)
                renewal_error = next(
                    (error for error in result if isinstance(error, AppError)),
                    None,
                )
            if acquired:
                try:
                    await self._release(session_id, owner)
                except AppError as error:
                    release_error = error
            return renewal_error or release_error
        finally:
            await self._unregister_turn()

    async def _wait_for_cleanup(
        self,
        cleanup: asyncio.Task[AppError | None],
    ) -> AppError | None:
        cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                cancellation = error
        if cancellation is not None:
            if not cleanup.cancelled():
                cleanup.exception()
            raise cancellation
        return cleanup.result()

    async def _acquire(self, session_id: str, owner: str, timeout_seconds: float) -> None:
        lease_key = self._keys(session_id).lease
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppError(CodigoError.SESION_OCUPADA, f"sesión ocupada: {session_id}")
            result = await self._eval(
                scripts.ACQUIRE,
                (lease_key,),
                owner,
                self._lease_milliseconds,
                timeout_seconds=remaining,
            )
            if int(cast(int, result)) == 1:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppError(CodigoError.SESION_OCUPADA, f"sesión ocupada: {session_id}")
            await asyncio.sleep(min(self._retry_interval_seconds, remaining))

    async def _renew_loop(
        self,
        session_id: str,
        owner: str,
        owner_task: asyncio.Task,
    ) -> None:
        interval = self._lease_milliseconds / 3_000
        try:
            while True:
                await asyncio.sleep(interval)
                lease_key = self._keys(session_id).lease
                result = await self._eval(
                    scripts.RENEW,
                    (lease_key,),
                    owner,
                    self._lease_milliseconds,
                )
                if int(cast(int, result)) != 1:
                    raise AppError(
                        CodigoError.SESION_LEASE_PERDIDO,
                        "ownership de sesión perdido",
                    )
        except asyncio.CancelledError:
            raise
        except AppError:
            owner_task.cancel()
            raise
        except Exception as error:
            failure = AppError(
                CodigoError.SESION_STORE_FALLO,
                f"renovación lanzó {type(error).__name__}",
            )
            owner_task.cancel()
            raise failure from error

    async def _load(self, session_id: str, owner: str) -> bytes | None:
        keys = self._keys(session_id)
        raw = await self._eval(
            scripts.LOAD,
            (keys.lease, keys.history),
            owner,
            self._ttl_milliseconds,
        )
        result = cast(list[object], raw)
        if not result or int(cast(int, result[0])) != 1:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")
        if len(result) == 1:
            return None
        return cast(bytes, result[1])

    async def _save(self, session_id: str, owner: str, payload: bytes) -> None:
        keys = self._keys(session_id)
        result = await self._eval(
            scripts.SAVE,
            (keys.lease, keys.history),
            owner,
            payload,
            self._ttl_milliseconds,
        )
        if int(cast(int, result)) != 1:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def _release(self, session_id: str, owner: str) -> None:
        lease_key = self._keys(session_id).lease
        result = await self._eval(scripts.RELEASE, (lease_key,), owner)
        if int(cast(int, result)) != 1:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def _eval(
        self,
        script: str,
        keys: tuple[str, ...],
        *arguments: _RedisArgument,
        timeout_seconds: float | None = None,
    ) -> object:
        self._ensure_open()
        timeout = min(timeout_seconds or self._io_timeout_seconds, self._io_timeout_seconds)
        try:
            async with asyncio.timeout(timeout):
                return await self._client.eval(script, len(keys), *keys, *arguments)
        except TimeoutError as error:
            raise AppError(
                CodigoError.SESION_STORE_FALLO,
                "Redis agotó el timeout de I/O",
            ) from error
        except RedisError as error:
            raise AppError(
                CodigoError.SESION_STORE_FALLO,
                f"Redis lanzó {type(error).__name__}",
            ) from error

    async def _register_turn(self) -> None:
        async with self._guard:
            self._ensure_open()
            self._active_turns += 1

    async def _unregister_turn(self) -> None:
        async with self._guard:
            self._active_turns -= 1

    async def _close_client(self) -> None:
        try:
            async with asyncio.timeout(self._io_timeout_seconds):
                await self._client.aclose()
        except TimeoutError as error:
            raise AppError(
                CodigoError.SESION_STORE_FALLO,
                "Redis no cerró su cliente dentro del timeout",
            ) from error
        except RedisError as error:
            raise AppError(
                CodigoError.SESION_STORE_FALLO,
                f"Redis lanzó {type(error).__name__} al cerrar",
            ) from error

    def _keys(self, session_id: str) -> _RedisKeys:
        digest = sha256(session_id.encode()).hexdigest()
        base = f"{self._key_prefix}:{{{digest}}}"
        return _RedisKeys(lease=f"{base}:lease", history=f"{base}:history")

    def _validate_turn(self, session_id: str, timeout_seconds: float) -> None:
        if not session_id or timeout_seconds <= 0:
            raise AppError(CodigoError.CONFIG_INVALIDA, "adquisición de sesión inválida")

    def _raise_renewal_failure(self, lease: "_Lease") -> None:
        failure = self._renewal_failure(lease.renewal)
        if failure is not None:
            if lease.owner_task.cancelling() != 1:
                raise asyncio.CancelledError
            lease.owner_task.uncancel()
            raise failure

    def _renewal_failure(self, renewal: asyncio.Task[None] | None) -> AppError | None:
        if renewal is None or not renewal.done() or renewal.cancelled():
            return None
        try:
            renewal.result()
        except AppError as error:
            return error
        return None

    def _ensure_open(self) -> None:
        if self._closed:
            raise AppError(CodigoError.RECURSO_CERRADO, "store Redis cerrado")


@dataclass(frozen=True, slots=True)
class _Lease:
    owner: str
    renewal: asyncio.Task[None]
    owner_task: asyncio.Task


def _milliseconds(seconds: float) -> int:
    value = int(seconds * 1_000)
    if value < 1:
        raise AppError(CodigoError.CONFIG_INVALIDA, "duración Redis demasiado pequeña")
    return value
