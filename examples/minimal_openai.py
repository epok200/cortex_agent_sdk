import asyncio

from pydantic_ai import Agent


async def main() -> None:
    agent = Agent("openai-responses:gpt-5.6-luna")
    async with agent:
        result = await agent.run("Responde únicamente: hola")
    print(result.output)


asyncio.run(main())
