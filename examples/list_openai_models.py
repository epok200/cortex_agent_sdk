import asyncio
import os

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.gateway import OpenAICompatibleGateway
from cortex_agent_sdk.openai import OpenAIEngine, OpenAIModel


def configured_gateway() -> OpenAICompatibleGateway | None:
    url = os.getenv("CORTEX_GATEWAY_URL")
    api_key = os.getenv("CORTEX_GATEWAY_API_KEY")
    if url is None and api_key is None:
        return None
    if url is None or api_key is None:
        raise AppError(
            CodigoError.CONFIG_INVALIDA,
            "el gateway requiere CORTEX_GATEWAY_URL y CORTEX_GATEWAY_API_KEY",
        )
    return OpenAICompatibleGateway(url=url, api_key=api_key)


async def list_models() -> tuple[OpenAIModel, ...]:
    gateway = configured_gateway()
    async with OpenAIEngine("gpt-5.6-luna", gateway=gateway) as engine:
        return await engine.models.list()


async def main() -> None:
    models = await list_models()
    for model in models:
        print(model.id)


if __name__ == "__main__":
    asyncio.run(main())
