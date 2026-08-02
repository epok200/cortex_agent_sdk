import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.sessions.models import SessionRecord
from cortex_agent_sdk.sessions.store import SessionLease


class LeaseKeeper:
    def __init__(self, lease: SessionLease) -> None:
        self._lease = lease
        self._fencing_token = lease.fencing_token
        self._heartbeat: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self.verify()
        interval = self._lease.renewal_interval_seconds
        if interval is not None and interval <= 0:
            raise AppError(CodigoError.CONFIG_INVALIDA, "intervalo de renovación inválido")
        if interval is not None:
            self._heartbeat = asyncio.create_task(self._renew_loop())

    async def load(self) -> SessionRecord | None:
        return await self.run(lambda: self._lease.load(self._fencing_token))

    async def save(self, record: SessionRecord) -> SessionRecord:
        return await self.run(lambda: self._lease.save(record, self._fencing_token))

    async def verify(self) -> None:
        self._raise_heartbeat_failure()
        await self._lease.verify_ownership(self._fencing_token)

    async def run[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        await self.verify()
        if self._heartbeat is None:
            return await operation()

        operation_task = asyncio.ensure_future(operation())
        try:
            done, _ = await asyncio.wait(
                (operation_task, self._heartbeat),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._heartbeat in done:
                operation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await operation_task
                self._raise_heartbeat_failure()
            return await operation_task
        finally:
            if not operation_task.done():
                operation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await operation_task

    async def aclose(self) -> None:
        if self._heartbeat is None:
            return
        self._heartbeat.cancel()
        with suppress(asyncio.CancelledError, AppError):
            await self._heartbeat

    async def _renew_loop(self) -> None:
        interval = self._lease.renewal_interval_seconds
        if interval is None:
            return
        try:
            while True:
                await asyncio.sleep(interval)
                await self._lease.renew(self._fencing_token)
        except asyncio.CancelledError:
            raise
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                CodigoError.SESION_LEASE_PERDIDO,
                f"renovación lanzó {type(error).__name__}",
            ) from error

    def _raise_heartbeat_failure(self) -> None:
        if self._heartbeat is not None and self._heartbeat.done():
            self._heartbeat.result()

