import asyncio
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.sessions.models import SessionRecord
from cortex_agent_sdk.sessions.store import SessionLease


@dataclass(slots=True)
class _Entry:
    record: SessionRecord
    expires_at: float


@dataclass(slots=True)
class _SessionLock:
    lock: asyncio.Lock
    users: int = 0


class _MemoryLease:
    def __init__(self, store: "MemorySessionStore", session_id: str, fencing_token: int) -> None:
        self._store = store
        self._session_id = session_id
        self._fencing_token = fencing_token
        self._active = True

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def fencing_token(self) -> int:
        return self._fencing_token

    @property
    def renewal_interval_seconds(self) -> float | None:
        return None

    async def verify_ownership(self, fencing_token: int) -> None:
        self._ensure_active(fencing_token)
        self._store._ensure_open()

    async def renew(self, fencing_token: int) -> None:
        await self.verify_ownership(fencing_token)

    async def load(self, fencing_token: int) -> SessionRecord | None:
        await self.verify_ownership(fencing_token)
        return await self._store._load(self._session_id)

    async def save(self, record: SessionRecord, fencing_token: int) -> SessionRecord:
        await self.verify_ownership(fencing_token)
        if record.session_id != self._session_id:
            raise AppError(CodigoError.SESION_INVALIDA, "session_id no coincide con el lease")
        return await self._store._save(record)

    def close(self) -> None:
        self._active = False

    def _ensure_active(self, fencing_token: int) -> None:
        if not self._active or fencing_token != self._fencing_token:
            raise AppError(CodigoError.SESION_LEASE_PERDIDO, "ownership de sesión perdido")


class MemorySessionStore:
    """Store local con exclusión por sesión, CAS, TTL y LRU."""

    def __init__(self, max_sessions: int = 1_000, ttl_seconds: float = 3_600) -> None:
        if max_sessions < 1 or ttl_seconds <= 0:
            raise AppError(CodigoError.CONFIG_INVALIDA, "límites de sesión inválidos")
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._locks: dict[str, _SessionLock] = {}
        self._guard = asyncio.Lock()
        self._closed = False
        self._next_fencing_token = 1

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

        async with self._guard:
            lock_entry = self._locks.setdefault(session_id, _SessionLock(asyncio.Lock()))
            lock_entry.users += 1
            fencing_token = self._next_fencing_token
            self._next_fencing_token += 1

        try:
            await asyncio.wait_for(lock_entry.lock.acquire(), timeout_seconds)
        except TimeoutError as error:
            await self._release_lock_reference(session_id, lock_entry)
            raise AppError(CodigoError.SESION_OCUPADA, f"sesión ocupada: {session_id}") from error
        except BaseException:
            await self._release_lock_reference(session_id, lock_entry)
            raise

        lease = _MemoryLease(self, session_id, fencing_token)
        try:
            yield lease
        finally:
            lease.close()
            lock_entry.lock.release()
            await self._release_lock_reference(session_id, lock_entry)

    async def reset(self, session_id: str, timeout_seconds: float = 5.0) -> None:
        async with self.acquire(session_id, timeout_seconds), self._guard:
            self._entries.pop(session_id, None)

    async def aclose(self) -> None:
        async with self._guard:
            self._closed = True
            self._entries.clear()
            self._locks.clear()

    async def _load(self, session_id: str) -> SessionRecord | None:
        async with self._guard:
            self._ensure_open()
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                del self._entries[session_id]
                return None
            entry.expires_at = time.monotonic() + self._ttl_seconds
            self._entries.move_to_end(session_id)
            return entry.record

    async def _save(self, record: SessionRecord) -> SessionRecord:
        async with self._guard:
            self._ensure_open()
            current = self._entries.get(record.session_id)
            current_version = current.record.version if current else 0
            if current_version != record.version:
                raise AppError(
                    CodigoError.SESION_CONFLICTO,
                    f"versión esperada {record.version}, vigente {current_version}",
                )

            saved = record.model_copy(
                update={
                    "version": record.version + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            expires_at = time.monotonic() + self._ttl_seconds
            self._entries[record.session_id] = _Entry(saved, expires_at)
            self._entries.move_to_end(record.session_id)
            while len(self._entries) > self._max_sessions:
                self._entries.popitem(last=False)
            return saved

    def _ensure_open(self) -> None:
        if self._closed:
            raise AppError(CodigoError.RECURSO_CERRADO, "store de sesiones cerrado")

    async def _release_lock_reference(
        self,
        session_id: str,
        lock_entry: _SessionLock,
    ) -> None:
        async with self._guard:
            lock_entry.users -= 1
            current = self._locks.get(session_id)
            if current is lock_entry and lock_entry.users == 0:
                del self._locks[session_id]

