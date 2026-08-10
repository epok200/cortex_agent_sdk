import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider


async def main() -> None:
    provider = OpenAIProvider(
        base_url=f"{os.environ['OPENAI_COMPATIBLE_URL'].rstrip('/')}/v1",
        api_key=os.environ["OPENAI_COMPATIBLE_API_KEY"],
    )
    provider.client.max_retries = 0
    model = OpenAIResponsesModel("gpt-5.6-luna", provider=provider)
    agent = Agent(model)

    async with agent:
        result = await agent.run("Responde únicamente: hola")
    print(result.output)


asyncio.run(main())
