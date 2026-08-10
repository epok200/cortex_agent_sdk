import asyncio

from pydantic_ai import Agent

from cortex_agent_sdk.capabilities import last_tool_result_fallback


async def main() -> None:
    agent = Agent(
        "openai-responses:gpt-5.6-luna",
        capabilities=[last_tool_result_fallback({"confirmar_agenda"})],
    )

    @agent.tool_plain
    def confirmar_agenda() -> str:
        """Confirma una operación de agenda."""
        return "Cita agendada para mañana a las 10:00."

    async with agent:
        result = await agent.run("Agenda una cita para mañana a las 10:00.")
    print(result.output)


asyncio.run(main())
