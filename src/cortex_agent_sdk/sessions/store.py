from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from cortex_agent_sdk.sessions.models import Session


@runtime_checkable
class SessionStore(Protocol):
    def turn(
        self,
        session_id: str,
        timeout_seconds: float = 5.0,
    ) -> AbstractAsyncContextManager[Session]: ...

    async def reset(self, session_id: str, timeout_seconds: float = 5.0) -> None: ...

    async def aclose(self) -> None: ...
