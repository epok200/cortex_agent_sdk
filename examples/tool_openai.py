import asyncio

from cortex_agent_sdk import Agent, final_answer
from cortex_agent_sdk.openai import OpenAIEngine


@final_answer
async def sumar(a: int, b: int) -> str:
    """Suma dos enteros."""
    return str(a + b)


async def main() -> None:
    instructions = "Para sumar, usa siempre la herramienta sumar."
    async with Agent(
        OpenAIEngine("gpt-5.6-luna"), instructions=instructions, tools=(sumar,)
    ) as agent:
        result = await agent.run("Suma 20 y 22.")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
