import asyncio

from pydantic_ai import Agent

from cortex_agent_sdk.sessions import MemorySessionStore


async def main() -> None:
    agent = Agent("openai-responses:gpt-5.6-luna")
    sessions = MemorySessionStore()

    async with agent, sessions:
        async with sessions.turn("demo") as session:
            result = await agent.run(
                "Recuerda que mi color favorito es verde.",
                message_history=session.messages,
                conversation_id=session.session_id,
            )
            session.replace(result.all_messages())

        async with sessions.turn("demo") as session:
            result = await agent.run(
                "¿Cuál es mi color favorito?",
                message_history=session.messages,
                conversation_id=session.session_id,
            )
            session.replace(result.all_messages())

    print(result.output)


asyncio.run(main())
