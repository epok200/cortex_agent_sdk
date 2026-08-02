import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from hashlib import sha256
from typing import Self, cast
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.redis import scripts
from cortex_agent_sdk.sessions.codec import SessionCodec
from cortex_agent_sdk.sessions.models import SessionRecord
from cortex_agent_sdk.sessions.store import SessionLease

type _RedisArgument = str | bytes | int | float


class _RedisLease:
    def __init__(
        self,
        store: "RedisSessionStore",
        session_id: str,
        owner: str,
        fencing_token: int,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._owner = owner
        self._fencing_token = fencing_token
        self._active = True

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def fencing_token(self) -> int:
        return self._fencing_token

    @property
    def renewal_interval_seconds(self) -> float:
        return self._store.renewal_interval_seconds

    async def verify_ownership(self, fencing_token: int) -> None:
        self._ensure_active(fencing_token)
        await self._store._verify(self._session_id, self._lease_value)

    async def renew(self, fencing_token: int) -> None:
        self._ensure_active(fencing_token)
        await self._store._renew(self._session_id, self._lease_value)

    async def load(self, fencing_token: int) -> SessionRecord | None:
        self._ensure_active(fencing_token)
        return await self._store._load(self._session_id, self._lease_value)

    async def save(self, record: SessionRecord, fencing_token: int) -> SessionRecord:
        self._ensure_active(fencing_token)
        if record.session_id != self._session_id:
            raise AppError(CodigoError.SESION_INVALIDA, "session_id no coincide con el lease")
        return await self._store._save(record, self._lease_value)

    def close(self) -> None:
        self._active = False

    @property
    def _lease_value(self) -> str:
        return f"{self._owner}:{self._fencing_token}"

    def _ensure_active(self, fencing_token: int) -> None:
        if not self._active or fencing_token != self._fencing_token:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")


class RedisSessionStore:
    """Store persistente con lease renovable y CAS atómico en Redis."""

    def __init__(
        self,
        url: str | None = None,
        *,
        client: Redis | None = None,
        key_prefix: str = "cortex:sessions",
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
        self._codec = SessionCodec()
        self._closed = False

    @property
    def renewal_interval_seconds(self) -> float:
        return self._lease_milliseconds / 3_000

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    @asynccontextmanager
    async def acquire(
        self,
        session_id: str,
        timeout_seconds: float,
    ) -> AsyncIterator[SessionLease]:
        self._ensure_open()
        if not session_id or timeout_seconds <= 0:
            raise AppError(CodigoError.CONFIG_INVALIDA, "adquisición de sesión inválida")

        owner = uuid4().hex
        fencing_token = await self._acquire(session_id, owner, timeout_seconds)
        lease = _RedisLease(self, session_id, owner, fencing_token)
        try:
            yield lease
        finally:
            lease.close()
            with suppress(AppError):
                await self._release(session_id, f"{owner}:{fencing_token}")

    async def reset(self, session_id: str, timeout_seconds: float = 5.0) -> None:
        async with self.acquire(session_id, timeout_seconds) as lease:
            result = await self._eval(
                scripts.RESET,
                self._keys(session_id)[:2],
                self._lease_value(lease),
            )
            if int(cast(int, result)) != 1:
                raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._owns_client:
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
        self._closed = True

    async def _acquire(self, session_id: str, owner: str, timeout_seconds: float) -> int:
        lease_key, _, fence_key = self._keys(session_id)
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppError(CodigoError.SESION_OCUPADA, f"sesión ocupada: {session_id}")
            result = await self._eval(
                scripts.ACQUIRE,
                (lease_key, fence_key),
                owner,
                self._lease_milliseconds,
                timeout_seconds=remaining,
            )
            fencing_token = int(cast(int, result))
            if fencing_token > 0:
                return fencing_token
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppError(CodigoError.SESION_OCUPADA, f"sesión ocupada: {session_id}")
            await asyncio.sleep(min(self._retry_interval_seconds, remaining))

    async def _verify(self, session_id: str, lease_value: str) -> None:
        lease_key, _, _ = self._keys(session_id)
        result = await self._eval(scripts.VERIFY, (lease_key,), lease_value)
        if int(cast(int, result)) != 1:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def _renew(self, session_id: str, lease_value: str) -> None:
        lease_key, _, _ = self._keys(session_id)
        result = await self._eval(
            scripts.RENEW,
            (lease_key,),
            lease_value,
            self._lease_milliseconds,
        )
        if int(cast(int, result)) != 1:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def _load(self, session_id: str, lease_value: str) -> SessionRecord | None:
        lease_key, session_key, _ = self._keys(session_id)
        raw = await self._eval(
            scripts.LOAD,
            (lease_key, session_key),
            lease_value,
            self._ttl_milliseconds,
        )
        result = cast(list[object], raw)
        if not result or int(cast(int, result[0])) != 1:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")
        if len(result) == 1:
            return None
        return self._codec.decode(cast(bytes, result[1]))

    async def _save(self, record: SessionRecord, lease_value: str) -> SessionRecord:
        saved = record.model_copy(
            update={
                "version": record.version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        lease_key, session_key, _ = self._keys(record.session_id)
        result = await self._eval(
            scripts.SAVE,
            (lease_key, session_key),
            lease_value,
            record.version,
            saved.version,
            self._codec.encode(saved),
            self._ttl_milliseconds,
        )
        status = int(cast(int, result))
        if status == 1:
            return saved
        if status == -1:
            raise AppError(CodigoError.SESION_CONFLICTO, "la versión de sesión cambió")
        raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def _release(self, session_id: str, lease_value: str) -> None:
        lease_key, _, _ = self._keys(session_id)
        await self._eval(scripts.RELEASE, (lease_key,), lease_value)

    async def _eval(
        self,
        script: str,
        keys: tuple[str, ...],
        *arguments: _RedisArgument,
        timeout_seconds: float | None = None,
    ) -> object:
        self._ensure_open()
        timeout = timeout_seconds or self._io_timeout_seconds
        try:
            async with asyncio.timeout(min(timeout, self._io_timeout_seconds)):
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

    def _keys(self, session_id: str) -> tuple[str, str, str]:
        digest = sha256(session_id.encode()).hexdigest()
        base = f"{self._key_prefix}:{{{digest}}}"
        return f"{base}:lease", f"{base}:record", f"{base}:fence"

    def _lease_value(self, lease: SessionLease) -> str:
        if not isinstance(lease, _RedisLease):
            raise AppError(CodigoError.SESION_INVALIDA, "lease de Redis inválido")
        return lease._lease_value

    def _ensure_open(self) -> None:
        if self._closed:
            raise AppError(CodigoError.RECURSO_CERRADO, "store Redis cerrado")


def _milliseconds(seconds: float) -> int:
    value = int(seconds * 1_000)
    if value < 1:
        raise AppError(CodigoError.CONFIG_INVALIDA, "duración Redis demasiado pequeña")
    return value

