import asyncio
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Self, cast
from uuid import uuid4

import asyncpg
from asyncpg import InterfaceError, Pool, PostgresError, Record

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.sessions.codec import SessionCodec
from cortex_agent_sdk.sessions.models import SessionRecord
from cortex_agent_sdk.sessions.store import SessionLease

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")


class _PostgresLease:
    def __init__(
        self,
        store: "PostgresSessionStore",
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
        await self._store._verify(self._session_id, self._owner, self._fencing_token)

    async def renew(self, fencing_token: int) -> None:
        self._ensure_active(fencing_token)
        await self._store._renew(self._session_id, self._owner, self._fencing_token)

    async def load(self, fencing_token: int) -> SessionRecord | None:
        self._ensure_active(fencing_token)
        return await self._store._load(self._session_id, self._owner, self._fencing_token)

    async def save(self, record: SessionRecord, fencing_token: int) -> SessionRecord:
        self._ensure_active(fencing_token)
        if record.session_id != self._session_id:
            raise AppError(CodigoError.SESION_INVALIDA, "session_id no coincide con el lease")
        return await self._store._save(record, self._owner, self._fencing_token)

    def close(self) -> None:
        self._active = False

    def _ensure_active(self, fencing_token: int) -> None:
        if not self._active or fencing_token != self._fencing_token:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")


class PostgresSessionStore:
    """Store persistente con lease renovable y CAS en Postgres."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: Pool | None = None,
        table_name: str = "cortex_agent_sessions",
        ttl_seconds: float = 86_400,
        lease_seconds: float = 30,
        retry_interval_seconds: float = 0.05,
        io_timeout_seconds: float = 10,
    ) -> None:
        if (dsn is None) == (pool is None):
            raise AppError(CodigoError.CONFIG_INVALIDA, "indica dsn o pool de Postgres")
        if _IDENTIFIER.fullmatch(table_name) is None:
            raise AppError(CodigoError.CONFIG_INVALIDA, "nombre de tabla inválido")
        durations = (ttl_seconds, lease_seconds, retry_interval_seconds, io_timeout_seconds)
        if min(durations) <= 0:
            raise AppError(CodigoError.CONFIG_INVALIDA, "configuración de Postgres inválida")

        self._dsn = dsn
        self._pool = pool
        self._owns_pool = pool is None
        self._table = f'"{table_name}"'
        self._ttl_seconds = ttl_seconds
        self._lease_seconds = lease_seconds
        self._retry_interval_seconds = retry_interval_seconds
        self._io_timeout_seconds = io_timeout_seconds
        self._pool_lock = asyncio.Lock()
        self._codec = SessionCodec()
        self._closed = False

    @property
    def renewal_interval_seconds(self) -> float:
        return self._lease_seconds / 3

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def setup(self) -> None:
        query = f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                session_id TEXT PRIMARY KEY,
                payload BYTEA,
                version BIGINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ,
                expires_at TIMESTAMPTZ,
                lease_owner TEXT,
                fencing_token BIGINT NOT NULL DEFAULT 0,
                lease_until TIMESTAMPTZ
            )
        """
        await self._execute(query)

    async def cleanup_expired(self, limit: int = 1_000) -> int:
        """Borra payloads vencidos sin eliminar su fencing counter."""
        if limit < 1:
            raise AppError(CodigoError.CONFIG_INVALIDA, "limit debe ser positivo")
        query = f"""
            WITH expired AS (
                SELECT session_id
                FROM {self._table}
                WHERE expires_at <= NOW()
                  AND (lease_until IS NULL OR lease_until <= NOW())
                ORDER BY expires_at
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE {self._table} AS sessions
            SET payload = NULL, version = 0, updated_at = NULL, expires_at = NULL
            FROM expired
            WHERE sessions.session_id = expired.session_id
        """
        status = await self._execute(query, limit)
        return int(status.rsplit(" ", 1)[-1])

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
        lease = _PostgresLease(self, session_id, owner, fencing_token)
        try:
            yield lease
        finally:
            lease.close()
            with suppress(AppError):
                await self._release(session_id, owner, fencing_token)

    async def reset(self, session_id: str, timeout_seconds: float = 5.0) -> None:
        async with self.acquire(session_id, timeout_seconds) as lease:
            if not isinstance(lease, _PostgresLease):
                raise AppError(CodigoError.SESION_INVALIDA, "lease de Postgres inválido")
            query = f"""
                UPDATE {self._table}
                SET payload = NULL, version = 0, updated_at = NULL, expires_at = NULL
                WHERE session_id = $1
                  AND lease_owner = $2
                  AND fencing_token = $3
                  AND lease_until > NOW()
                RETURNING fencing_token
            """
            result = await self._fetchval(
                query,
                session_id,
                lease._owner,
                lease.fencing_token,
            )
            if result is None:
                raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._owns_pool and self._pool is not None:
            try:
                async with asyncio.timeout(self._io_timeout_seconds):
                    await self._pool.close()
            except TimeoutError as error:
                raise AppError(
                    CodigoError.SESION_STORE_FALLO,
                    "Postgres no cerró su pool dentro del timeout",
                ) from error
            except (InterfaceError, PostgresError, OSError) as error:
                raise _store_error(error) from error
        self._closed = True

    async def _acquire(self, session_id: str, owner: str, timeout_seconds: float) -> int:
        query = f"""
            INSERT INTO {self._table} AS sessions (
                session_id, lease_owner, fencing_token, lease_until
            )
            VALUES ($1, $2, 1, NOW() + $3 * INTERVAL '1 second')
            ON CONFLICT (session_id) DO UPDATE
            SET lease_owner = EXCLUDED.lease_owner,
                fencing_token = sessions.fencing_token + 1,
                lease_until = EXCLUDED.lease_until
            WHERE sessions.lease_until IS NULL OR sessions.lease_until <= NOW()
            RETURNING fencing_token
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppError(CodigoError.SESION_OCUPADA, f"sesión ocupada: {session_id}")
            result = await self._fetchval(
                query,
                session_id,
                owner,
                self._lease_seconds,
                timeout_seconds=remaining,
            )
            if result is not None:
                return cast(int, result)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppError(CodigoError.SESION_OCUPADA, f"sesión ocupada: {session_id}")
            await asyncio.sleep(min(self._retry_interval_seconds, remaining))

    async def _verify(self, session_id: str, owner: str, fencing_token: int) -> None:
        query = f"""
            SELECT EXISTS(
                SELECT 1 FROM {self._table}
                WHERE session_id = $1
                  AND lease_owner = $2
                  AND fencing_token = $3
                  AND lease_until > NOW()
            )
        """
        if not await self._fetchval(query, session_id, owner, fencing_token):
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def _renew(self, session_id: str, owner: str, fencing_token: int) -> None:
        query = f"""
            UPDATE {self._table}
            SET lease_until = NOW() + $4 * INTERVAL '1 second'
            WHERE session_id = $1
              AND lease_owner = $2
              AND fencing_token = $3
              AND lease_until > NOW()
            RETURNING fencing_token
        """
        result = await self._fetchval(
            query,
            session_id,
            owner,
            fencing_token,
            self._lease_seconds,
        )
        if result is None:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def _load(
        self,
        session_id: str,
        owner: str,
        fencing_token: int,
    ) -> SessionRecord | None:
        query = f"""
            UPDATE {self._table}
            SET payload = CASE
                    WHEN expires_at <= NOW() THEN NULL
                    ELSE payload
                END,
                version = CASE
                    WHEN expires_at <= NOW() THEN 0
                    ELSE version
                END,
                updated_at = CASE
                    WHEN expires_at <= NOW() THEN NULL
                    ELSE updated_at
                END,
                expires_at = CASE
                    WHEN payload IS NULL OR expires_at <= NOW() THEN NULL
                    ELSE NOW() + $4 * INTERVAL '1 second'
                END
            WHERE session_id = $1
              AND lease_owner = $2
              AND fencing_token = $3
              AND lease_until > NOW()
            RETURNING payload
        """
        row = await self._fetchrow(
            query,
            session_id,
            owner,
            fencing_token,
            self._ttl_seconds,
        )
        if row is None:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")
        payload = row["payload"]
        if payload is None:
            return None
        return self._codec.decode(bytes(payload))

    async def _save(
        self,
        record: SessionRecord,
        owner: str,
        fencing_token: int,
    ) -> SessionRecord:
        saved = record.model_copy(
            update={
                "version": record.version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        query = f"""
            UPDATE {self._table}
            SET payload = $5,
                version = $6,
                updated_at = NOW(),
                expires_at = NOW() + $7 * INTERVAL '1 second'
            WHERE session_id = $1
              AND lease_owner = $2
              AND fencing_token = $3
              AND lease_until > NOW()
              AND version = $4
            RETURNING version
        """
        result = await self._fetchval(
            query,
            record.session_id,
            owner,
            fencing_token,
            record.version,
            self._codec.encode(saved),
            saved.version,
            self._ttl_seconds,
        )
        if result is not None:
            return saved
        await self._raise_save_failure(record, owner, fencing_token)
        raise AssertionError("save failure no traducido")

    async def _raise_save_failure(
        self,
        record: SessionRecord,
        owner: str,
        fencing_token: int,
    ) -> None:
        query = f"""
            SELECT version, lease_owner, fencing_token, lease_until > NOW() AS lease_active
            FROM {self._table}
            WHERE session_id = $1
        """
        row = await self._fetchrow(query, record.session_id)
        owns_lease = (
            row is not None
            and row["lease_owner"] == owner
            and row["fencing_token"] == fencing_token
            and row["lease_active"]
        )
        if owns_lease:
            raise AppError(CodigoError.SESION_CONFLICTO, "la versión de sesión cambió")
        raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")

    async def _release(self, session_id: str, owner: str, fencing_token: int) -> None:
        query = f"""
            UPDATE {self._table}
            SET lease_owner = NULL, lease_until = NULL
            WHERE session_id = $1 AND lease_owner = $2 AND fencing_token = $3
        """
        await self._execute(query, session_id, owner, fencing_token)

    async def _get_pool(self) -> Pool:
        self._ensure_open()
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                try:
                    self._pool = await asyncpg.create_pool(dsn=cast(str, self._dsn))
                except (InterfaceError, PostgresError, OSError) as error:
                    raise _store_error(error) from error
            return self._pool

    async def _fetchval(
        self,
        query: str,
        *arguments: object,
        timeout_seconds: float | None = None,
    ) -> object:
        timeout = timeout_seconds or self._io_timeout_seconds
        try:
            async with asyncio.timeout(min(timeout, self._io_timeout_seconds)):
                return await (await self._get_pool()).fetchval(query, *arguments)
        except TimeoutError as error:
            raise AppError(
                CodigoError.SESION_STORE_FALLO,
                "Postgres agotó el timeout de I/O",
            ) from error
        except (InterfaceError, PostgresError, OSError) as error:
            raise _store_error(error) from error

    async def _fetchrow(
        self,
        query: str,
        *arguments: object,
        timeout_seconds: float | None = None,
    ) -> Record | None:
        timeout = timeout_seconds or self._io_timeout_seconds
        try:
            async with asyncio.timeout(min(timeout, self._io_timeout_seconds)):
                return await (await self._get_pool()).fetchrow(query, *arguments)
        except TimeoutError as error:
            raise AppError(
                CodigoError.SESION_STORE_FALLO,
                "Postgres agotó el timeout de I/O",
            ) from error
        except (InterfaceError, PostgresError, OSError) as error:
            raise _store_error(error) from error

    async def _execute(
        self,
        query: str,
        *arguments: object,
        timeout_seconds: float | None = None,
    ) -> str:
        timeout = timeout_seconds or self._io_timeout_seconds
        try:
            async with asyncio.timeout(min(timeout, self._io_timeout_seconds)):
                return await (await self._get_pool()).execute(query, *arguments)
        except TimeoutError as error:
            raise AppError(
                CodigoError.SESION_STORE_FALLO,
                "Postgres agotó el timeout de I/O",
            ) from error
        except (InterfaceError, PostgresError, OSError) as error:
            raise _store_error(error) from error

    def _ensure_open(self) -> None:
        if self._closed:
            raise AppError(CodigoError.RECURSO_CERRADO, "store Postgres cerrado")


def _store_error(error: BaseException) -> AppError:
    return AppError(
        CodigoError.SESION_STORE_FALLO,
        f"Postgres lanzó {type(error).__name__}",
    )

