from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from cortex_agent_sdk.engine import Usage
from cortex_agent_sdk.history.models import Turn
from cortex_agent_sdk.sessions.lease import LeaseKeeper
from cortex_agent_sdk.sessions.models import SessionRecord


class AgentOptions(BaseModel):
    """Límites del loop, hooks y sesiones."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(default=8, ge=1)
    max_history_turns: int = Field(default=20, ge=1)
    max_consecutive_tool_failures: int = Field(default=3, ge=1)
    provider_timeout_seconds: float = Field(default=120.0, gt=0)
    tool_timeout_seconds: float = Field(default=30.0, gt=0)
    hook_timeout_seconds: float = Field(default=10.0, gt=0)
    session_lock_timeout_seconds: float = Field(default=5.0, gt=0)
    shutdown_timeout_seconds: float = Field(default=120.0, gt=0)


@dataclass(slots=True)
class RunState:
    history: list[Turn]
    record: SessionRecord | None
    lease_keeper: LeaseKeeper | None
    instructions: str
    usage: Usage = field(default_factory=Usage)
    raw_responses: list[object] = field(default_factory=list)
    tool_calls: int = 0
    consecutive_tool_failures: int = 0
    last_text: str | None = None
    structured: BaseModel | None = None
    effective_model: str | None = None

