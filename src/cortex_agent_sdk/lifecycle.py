import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from cortex_agent_sdk.errores import AppError, CodigoError


class AgentLifecycle:
    """Coordina turnos activos y cierre ordenado del agente."""

    def __init__(self) -> None:
        self._active_runs = 0
        self._accepting_runs = True
        self._closed = False
        self._guard = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._drained = asyncio.Event()
        self._drained.set()

    async def ensure_open(self) -> None:
        async with self._guard:
            self._ensure_accepting()

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        async with self._guard:
            self._ensure_accepting()
            self._active_runs += 1
            self._drained.clear()
        try:
            yield
        finally:
            async with self._guard:
                self._active_runs -= 1
                if self._active_runs == 0:
                    self._drained.set()

    async def close(
        self,
        close_resources: Callable[[], Awaitable[None]],
        timeout_seconds: float,
    ) -> None:
        async with self._close_lock:
            async with self._guard:
                if self._closed:
                    return
                self._accepting_runs = False

            try:
                async with asyncio.timeout(timeout_seconds):
                    await self._drained.wait()
                    await close_resources()
            except TimeoutError as error:
                raise AppError(
                    CodigoError.RUNTIME_CIERRE_TIMEOUT,
                    "el agente no completó el drenado y cierre dentro del timeout",
                ) from error

            async with self._guard:
                self._closed = True

    def _ensure_accepting(self) -> None:
        if self._closed:
            raise AppError(CodigoError.RECURSO_CERRADO, "Agent cerrado")
        if not self._accepting_runs:
            raise AppError(CodigoError.RECURSO_CERRADO, "Agent en cierre ordenado")

