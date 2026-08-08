"""Ejemplo de approval estático/dinámico, serialización y resume."""

import asyncio
from collections.abc import Mapping

from cortex_agent_sdk import Agent, PendingRun, ToolApprovalContext, tool
from cortex_agent_sdk.openai import OpenAIEngine


@tool(needs_approval=True)
async def eliminar_evento(event_id: str) -> str:
    """Elimina un evento; siempre requiere autorización humana."""
    return f"Evento {event_id} eliminado."


def requiere_aprobacion_masiva(
    _context: ToolApprovalContext,
    arguments: Mapping[str, object],
    _call_id: str,
) -> bool:
    """Pide aprobación sólo cuando la call intenta eliminar más de tres eventos."""
    cantidad = arguments.get("cantidad")
    return isinstance(cantidad, int) and cantidad > 3


@tool(needs_approval=requiere_aprobacion_masiva)
async def eliminar_eventos(cantidad: int) -> str:
    """Elimina varios eventos; operaciones pequeñas pueden ejecutarse sin pausa."""
    return f"Se eliminaron {cantidad} eventos."


async def resolver_approvals(agent: Agent, pending: PendingRun):
    """CLI mínima: decide cada interruption y reanuda hasta que el run pueda continuar."""
    while pending.unresolved:
        for interruption in pending.unresolved:
            print(
                f"Tool: {interruption.tool_name} · "
                f"args={dict(interruption.arguments)}"
            )
            answer = await asyncio.to_thread(input, "¿Aprobar? [s/N]: ")
            if answer.strip().lower() == "s":
                pending.approve(interruption.call_id)
            else:
                pending.reject(
                    interruption.call_id,
                    message="El usuario rechazó esta operación.",
                )

    # El snapshot puede cruzar un request, proceso o almacenamiento externo.
    restored = PendingRun.from_json(pending.to_json())
    return await agent.resume(restored)


async def main() -> None:
    async with Agent(
        OpenAIEngine("gpt-5.6-luna"),
        instructions="Gestiona la agenda usando las tools disponibles.",
        tools=(eliminar_evento, eliminar_eventos),
    ) as agent:
        result = await agent.run("Elimina 5 eventos viejos de mi agenda.")

        while result.pending_run is not None:
            result = await resolver_approvals(agent, result.pending_run)

    print(result.reason)
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
