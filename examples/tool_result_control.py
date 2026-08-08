"""Ejemplo de los tres modos de resultado y una policy post-tool."""

import asyncio

from cortex_agent_sdk import (
    CONTINUE,
    Agent,
    FinalOutput,
    ToolResultContext,
    fallback_answer,
    final_answer,
    tool,
)
from cortex_agent_sdk.openai import OpenAIEngine


@tool
async def calcular_total(precio: float, cantidad: int) -> dict[str, float]:
    """Calcula un total y devuelve el dato al modelo para que siga razonando."""
    return {"total": precio * cantidad}


@fallback_answer
async def consultar_estado(orden_id: str) -> str:
    """Consulta una orden; su texto puede rescatar un cierre limpio sin respuesta del modelo."""
    return f"La orden {orden_id} sigue en preparación."


@tool
async def registrar_pago(orden_id: str) -> dict[str, object]:
    """Registra un pago ficticio para mostrar una decisión terminal por policy."""
    return {"ok": True, "orden_id": orden_id}


@final_answer
async def respuesta_directa(mensaje: str) -> str:
    """Devuelve una respuesta terminal sin pedir otro turno al modelo."""
    return mensaje


def finalizar_despues_del_pago(context: ToolResultContext):
    """Sólo el resultado de registrar_pago termina dinámicamente este run."""
    if any(
        outcome.call.name == "registrar_pago" and not outcome.failed
        for outcome in context.outcomes
    ):
        return FinalOutput("Pago registrado correctamente.")
    return CONTINUE


async def main() -> None:
    instructions = (
        "Usa las tools cuando sean necesarias. "
        "Puedes encadenar varias antes de responder al usuario."
    )
    async with Agent(
        OpenAIEngine("gpt-5.6-luna"),
        instructions=instructions,
        tools=(calcular_total, consultar_estado, registrar_pago, respuesta_directa),
        tool_result_policy=finalizar_despues_del_pago,
    ) as agent:
        result = await agent.run("Consulta la orden A-42 y dime su estado.")

    print(result.reason)
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
