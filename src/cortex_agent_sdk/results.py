from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from cortex_agent_sdk.control import PendingRun, ToolApproval
from cortex_agent_sdk.engine import Usage
from cortex_agent_sdk.history.models import Turn


class AgentExitReason(StrEnum):
    COMPLETED = "completed"
    FINAL_ANSWER = "final_answer"
    FALLBACK_ANSWER = "fallback_answer"
    TOOL_POLICY = "tool_policy"
    APPROVAL_REQUIRED = "approval_required"
    MAX_STEPS = "max_steps"
    TOOL_FAILURE_LIMIT = "tool_failure_limit"


@dataclass(frozen=True, slots=True)
class AgentResult:
    text: str | None
    reason: AgentExitReason
    usage: Usage
    steps: int
    tool_calls: int
    provider: str
    model: str
    history: tuple[Turn, ...]
    raw_responses: tuple[object, ...]
    structured: BaseModel | None = None
    interruptions: tuple[ToolApproval, ...] = ()
    pending_run: PendingRun | None = None
