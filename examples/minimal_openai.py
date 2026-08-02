import asyncio

from cortex_agent_sdk import Agent
from cortex_agent_sdk.openai import OpenAIEngine, OpenAIOptions


async def main() -> None:
    options = OpenAIOptions(max_output_tokens=128, reasoning_effort="low")
    async with Agent(OpenAIEngine("gpt-5.6-luna", options=options)) as agent:
        result = await agent.run("Responde únicamente: hola")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
