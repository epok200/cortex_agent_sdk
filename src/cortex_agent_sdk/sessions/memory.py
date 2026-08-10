import asyncio
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Self

from pydantic_ai.messages import ModelMessage

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.sessions.codec import decode_messages, encode_messages
from cortex_agent_sdk.sessions.models import ActiveTurn, Session


@dataclass(slots=True)
class _Entry:
    payload: bytes
    active_turn: ActiveTurn | None
    expires_at: float


@dataclass(slots=True)
class _SessionLock:
    lock: asyncio.Lock
    users: int = 0


@dataclass(slots=True)
class _TurnOwner:
    active: bool = True


class MemorySessionStore:
    """Sesiones locales con exclusión por turno, TTL y LRU."""

    def __init__(self, max_sessions: int = 1_000, ttl_seconds: float = 3_600) -> None:
        if max_sessions < 1 or ttl_seconds <= 0:
            raise AppError(CodigoError.CONFIG_INVALIDA, "límites de sesión inválidos")
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._locks: dict[str, _SessionLock] = {}
        self._guard = asyncio.Lock()
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
        async with self._lock(session_id, timeout_seconds):
            entry = await self._load(session_id)
            if entry is not None and entry.active_turn is not None:
                raise AppError(
                    CodigoError.SESION_RECUPERACION_REQUERIDA,
                    f"turno incompleto en sesión {session_id}",
                )
            messages = decode_messages(entry.payload) if entry is not None else []
            owner = _TurnOwner()
            session = Session(
                session_id,
                messages,
                _start_tool_turn=partial(self._mark_active, session_id, owner),
                _save_checkpoint=partial(self._checkpoint, session_id, owner),
            )
            try:
                yield session
                if session.changed:
                    await self._save(session_id, session.messages)
            finally:
                owner.active = False
                session._detach()
                await self._refresh_active(session_id)

    async def reset(self, session_id: str, timeout_seconds: float = 5.0) -> None:
        async with self._lock(session_id, timeout_seconds), self._guard:
            self._entries.pop(session_id, None)

    async def aclose(self) -> None:
        async with self._guard:
            if self._closed:
                return
            if self._locks:
                raise AppError(CodigoError.SESION_OCUPADA, "hay turnos de sesión activos")
            self._closed = True
            self._entries.clear()

    async def _load(self, session_id: str) -> _Entry | None:
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
            return entry

    async def _save(self, session_id: str, messages: list[ModelMessage]) -> None:
        payload = encode_messages(messages)
        async with self._guard:
            self._ensure_open()
            current = self._entries.get(session_id)
            active_turn = current.active_turn if current is not None else None
            self._write_entry(session_id, payload, active_turn)

    async def _mark_active(
        self,
        session_id: str,
        owner: _TurnOwner,
        active_turn: ActiveTurn,
    ) -> None:
        async with self._guard:
            self._ensure_open()
            if not owner.active:
                raise AppError(CodigoError.SESION_INVALIDA, "sesión fuera de su turno")
            current = self._entries.get(session_id)
            if current is not None and current.active_turn is not None:
                raise AppError(CodigoError.SESION_INVALIDA, "la sesión ya tiene un turno activo")
            payload = current.payload if current is not None else encode_messages([])
            self._write_entry(session_id, payload, active_turn)

    async def _checkpoint(
        self,
        session_id: str,
        owner: _TurnOwner,
        messages: Sequence[ModelMessage],
    ) -> None:
        payload = encode_messages(messages)
        async with self._guard:
            self._ensure_open()
            if not owner.active:
                raise AppError(CodigoError.SESION_INVALIDA, "sesión fuera de su turno")
            self._write_entry(session_id, payload, None)

    async def _refresh_active(self, session_id: str) -> None:
        async with self._guard:
            entry = self._entries.get(session_id)
            if entry is not None and entry.active_turn is not None:
                entry.expires_at = time.monotonic() + self._ttl_seconds

    def _write_entry(
        self,
        session_id: str,
        payload: bytes,
        active_turn: ActiveTurn | None,
    ) -> None:
        now = time.monotonic()
        expires_at = now + self._ttl_seconds
        self._entries[session_id] = _Entry(payload, active_turn, expires_at)
        self._entries.move_to_end(session_id)
        for key, entry in tuple(self._entries.items()):
            if entry.expires_at <= now and key not in self._locks:
                del self._entries[key]
        while len(self._entries) > self._max_sessions:
            evictable = next(
                (
                    key
                    for key, entry in self._entries.items()
                    if key != session_id
                    and key not in self._locks
                    and entry.active_turn is None
                ),
                None,
            )
            if evictable is None:
                break
            del self._entries[evictable]

    @asynccontextmanager
    async def _lock(
        self,
        session_id: str,
        timeout_seconds: float,
    ) -> AsyncIterator[None]:
        if not session_id or timeout_seconds <= 0:
            raise AppError(CodigoError.CONFIG_INVALIDA, "adquisición de sesión inválida")
        async with self._guard:
            self._ensure_open()
            entry = self._locks.setdefault(session_id, _SessionLock(asyncio.Lock()))
            entry.users += 1

        try:
            await asyncio.wait_for(entry.lock.acquire(), timeout_seconds)
        except TimeoutError as error:
            await self._release_reference(session_id, entry)
            raise AppError(CodigoError.SESION_OCUPADA, f"sesión ocupada: {session_id}") from error
        except BaseException:
            await self._release_reference(session_id, entry)
            raise

        try:
            yield
        finally:
            entry.lock.release()
            await self._release_reference(session_id, entry)

    async def _release_reference(self, session_id: str, entry: _SessionLock) -> None:
        async with self._guard:
            entry.users -= 1
            if self._locks.get(session_id) is entry and entry.users == 0:
                del self._locks[session_id]

    def _ensure_open(self) -> None:
        if self._closed:
            raise AppError(CodigoError.RECURSO_CERRADO, "store de sesiones cerrado")
