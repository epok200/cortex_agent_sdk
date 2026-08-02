from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from cortex_agent_sdk.sessions.models import SessionRecord


@runtime_checkable
class SessionLease(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def fencing_token(self) -> int: ...

    @property
    def renewal_interval_seconds(self) -> float | None: ...

    async def verify_ownership(self, fencing_token: int) -> None: ...

    async def renew(self, fencing_token: int) -> None: ...

    async def load(self, fencing_token: int) -> SessionRecord | None: ...

    async def save(self, record: SessionRecord, fencing_token: int) -> SessionRecord: ...


@runtime_checkable
class SessionStore(Protocol):
    def acquire(
        self,
        session_id: str,
        timeout_seconds: float,
    ) -> AbstractAsyncContextManager[SessionLease]: ...

    async def reset(self, session_id: str, timeout_seconds: float = 5.0) -> None: ...

    async def aclose(self) -> None: ...

