from cortex_agent_sdk.tools.models import ToolFunction


def tool(function: ToolFunction) -> ToolFunction:
    """Marca un callable async para registrarlo en un Agent."""
    function.__cortex_tool__ = True  # type: ignore[attr-defined]
    return function


def final_answer(function: ToolFunction) -> ToolFunction:
    """Marca una tool cuyo str exitoso termina el turno."""
    function.__cortex_tool__ = True  # type: ignore[attr-defined]
    function.__cortex_final_answer__ = True  # type: ignore[attr-defined]
    return function

