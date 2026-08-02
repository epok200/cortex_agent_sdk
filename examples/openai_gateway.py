import asyncio
import os

from cortex_agent_sdk import Agent
from cortex_agent_sdk.gateway import OpenAICompatibleGateway
from cortex_agent_sdk.openai import OpenAIEngine, OpenAIOptions


async def main() -> None:
    gateway = OpenAICompatibleGateway(
        url=os.environ["CORTEX_GATEWAY_URL"],
        api_key=os.environ["CORTEX_GATEWAY_API_KEY"],
    )
    options = OpenAIOptions(max_output_tokens=128, max_retries=0)
    engine = OpenAIEngine("gpt-5.6-luna", gateway=gateway, options=options)
    async with Agent(engine) as agent:
        result = await agent.run("Responde únicamente: hola")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
