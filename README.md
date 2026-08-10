# Cortex Agent SDK

Cortex agrega sesiones Memory y Redis, junto con capacidades puntuales que Pydantic AI no incluye,
a agentes multiprovider construidos directamente con Pydantic AI.

No implementa otro loop, otra capa de tools ni otra API de agentes. Pydantic AI conserva el control
de providers, modelos, tools, tipado, `RunContext`, hooks, límites, approvals, outputs, usage e
historial. Cortex aporta persistencia conversacional con un turno activo por sesión y capacidades
opcionales construidas sobre sus hooks públicos.

> Cortex Agent SDK está en alfa. La API puede cambiar antes de la versión `1.0.0`.

## Requisitos

- Python `>=3.13`.
- Pydantic AI `>=2.27,<3`.

## Instalación

Solo sesiones en memoria y capabilities:

```bash
uv add cortex-agent-sdk
```

OpenAI y sesiones en memoria:

```bash
uv add "cortex-agent-sdk[openai]"
```

OpenAI y Redis:

```bash
uv add "cortex-agent-sdk[openai,redis]"
```

Google se instala con el extra `google`. Cada producto elige únicamente sus providers. Cortex no
implementa adapters paralelos ni ofrece un extra que los instale todos.

## Uso

El agente es el `Agent` nativo de Pydantic AI. El store entrega el historial bajo exclusión y lo
guarda cuando `session.replace(...)` marca un resultado completo.
El siguiente ejemplo requiere el extra `openai`.

```python
import asyncio

from pydantic_ai import Agent

from cortex_agent_sdk.sessions import MemorySessionStore


async def main() -> None:
    agent = Agent("openai-responses:gpt-5.6-luna")
    sessions = MemorySessionStore()

    async with agent, sessions:
        async with sessions.turn("usuario:42") as session:
            result = await agent.run(
                "Recuerda que mi color favorito es verde.",
                message_history=session.messages,
                conversation_id=session.session_id,
            )
            session.replace(result.all_messages())

    print(result.output)


asyncio.run(main())
```

`replace()` es explícito por diseño:

- Si no se llama, el store no modifica el historial.
- Si el bloque termina con una excepción, el store no guarda el reemplazo.
- Si guardar falla, la excepción se propaga.

## Redis

```python
from cortex_agent_sdk.redis import RedisSessionStore

sessions = RedisSessionStore(
    "redis://localhost:6379/0",
    key_prefix="mi-producto:sesiones:v1",
    ttl_seconds=86_400,
)
```

Redis mantiene un lease renovable durante todo el turno. Mientras conserva el lease, dos procesos no
pueden usar la misma sesión al mismo tiempo y las sesiones distintas siguen siendo concurrentes. Si
la renovación falla, Cortex interrumpe el turno propietario y no guarda como owner obsoleto. El
historial se serializa con `ModelMessagesTypeAdapter`, el formato público de Pydantic AI.

Al cambiar desde el runtime anterior, usa un prefix nuevo. Los formatos no son compatibles y Cortex
no intenta convertir el historial legacy.

## Endpoint compatible con OpenAI

Pydantic AI puede conectarse directamente. Para un endpoint que no debe reintentar peticiones,
configura el provider una vez:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    base_url="https://example.com/v1",
    api_key="...",
)
provider.client.max_retries = 0
model = OpenAIResponsesModel("gpt-5.6-luna", provider=provider)
agent = Agent(model)
```

El context manager de `Agent` administra el transporte del provider.

## Fallback del resultado de una tool

Pydantic AI reintenta cuando un modelo termina sin texto. Para tools cuyo resultado ya es una
respuesta completa, Cortex puede reutilizar el último resultado exitoso del mismo run:

```python
from pydantic_ai import Agent

from cortex_agent_sdk.capabilities import last_tool_result_fallback

agent = Agent(
    "openai-responses:gpt-5.6-luna",
    capabilities=[last_tool_result_fallback({"confirmar_agenda"})],
)
```

La aplicación conserva la decisión sobre las tools elegibles. La capacidad no usa resultados
fallidos, vacíos ni pertenecientes a otro run, y no reemplaza texto o nuevas llamadas del modelo.

## Migración desde el runtime anterior

| Antes | Ahora |
|---|---|
| `cortex_agent_sdk.Agent` | `pydantic_ai.Agent` |
| `OpenAIEngine` | `OpenAIResponsesModel` + `OpenAIProvider` |
| `OpenAICompatibleGateway` | `OpenAIProvider(base_url=..., api_key=...)` |
| `OpenAIOptions` | `OpenAIResponsesModelSettings` y argumentos de `Agent.run` |
| `@tool` e `Injected` | tools nativas + `RunContext[Deps]` |
| `ToolBinding` | `FunctionToolset` o tools preparadas por run |
| `AgentHooks` | `pydantic_ai.capabilities.Hooks` |
| `turn_finished` | `Hooks(after_run=...)` |
| `history_transform` | `Hooks(before_model_request=...)` |
| `fallback_answer` | `last_tool_result_fallback(...)` opcional |
| `AgentOptions` | `UsageLimits`, settings del modelo y argumentos de `Agent` |
| `AgentResult.text` | `AgentRunResult.output` |
| `SessionStore.acquire` | `SessionStore.turn` + `Session.replace` |
| `agent.reset_session(id)` | `store.reset(id)` |

No se ofrece una capa de compatibilidad. Mantenerla volvería a duplicar la API y el runtime de
Pydantic AI.

## Superficie pública

- `cortex_agent_sdk.sessions.Session`
- `cortex_agent_sdk.sessions.SessionStore`
- `cortex_agent_sdk.sessions.MemorySessionStore`
- `cortex_agent_sdk.redis.RedisSessionStore`
- `cortex_agent_sdk.capabilities.last_tool_result_fallback`
- `cortex_agent_sdk.errores.AppError`
- `cortex_agent_sdk.errores.CodigoError`
- `cortex_agent_sdk.errores.Severidad`

El loop, tools, hooks, approvals, modelos y resultados se importan desde `pydantic_ai`.

## Licencia

Apache License 2.0.
