from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIOptions(BaseModel):
    """Opciones cohesionadas de Responses API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_output_tokens: int = Field(default=1_024, gt=0)
    timeout_seconds: float = Field(default=90.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    parallel_tool_calls: bool = False
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    reasoning_context: Literal["auto", "current_turn", "all_turns"] | None = None
    text_verbosity: Literal["low", "medium", "high"] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)

