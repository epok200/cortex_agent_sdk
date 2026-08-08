# Cortex Agent SDK

SDK async y multiproveedor para construir agentes con una API pequeña y control explícito del loop,
las tools, el historial y el ciclo de vida.

> Cortex Agent SDK está en alfa. La API puede cambiar antes de la versión `1.0.0`.

## Requisitos

- Python `>=3.13`.
- Una credencial del provider elegido.

## Instalación

OpenAI Responses:

```bash
uv add "cortex-agent-sdk[openai]"
```

También puede instalarse con `pip`:

```bash
python -m pip install "cortex-agent-sdk[openai]"
```

Extras disponibles: `openai`, `gateway`, `redis`, `postgres` y `all`.

## Uso mínimo

El SDK oficial de OpenAI lee `OPENAI_API_KEY` del entorno.

```python
import asyncio

from cortex_agent_sdk import Agent
from cortex_agent_sdk.openai import OpenAIEngine


async def main() -> None:
    async with Agent(OpenAIEngine("gpt-5.6-luna")) as agent:
        result = await agent.run("Responde únicamente: hola")
    print(result.text)


asyncio.run(main())
```

## Tools

Cortex conserva tres semánticas explícitas para el resultado de una tool:

- `@tool`: ejecuta y devuelve el resultado al modelo para continuar el loop.
- `@fallback_answer`: continúa el loop y conserva un resultado de respaldo para un cierre limpio sin
  texto visible.
- `@final_answer`: un `str` exitoso termina el run sin pedir otro turno al modelo.

Las tres variantes aceptan contratos inferidos por type hints o un `BaseModel` mediante
`input_model=...`. También pueden declarar `needs_approval=True` o un predicate sync/async para
human-in-the-loop por llamada.

`Agent(tool_result_policy=...)` queda como mecanismo opcional cuando la decisión de continuar o
terminar depende del resultado de una tanda de tools y no de una tool fija.

## Ejemplos

Los ejemplos ejecutables son la referencia práctica de uso:

- [`examples/minimal_openai.py`](examples/minimal_openai.py): ejecución mínima con OpenAI Responses.
- [`examples/tool_openai.py`](examples/tool_openai.py): `@final_answer` sencillo.
- [`examples/tool_result_control.py`](examples/tool_result_control.py): `tool`, `fallback_answer`,
  `final_answer` y `tool_result_policy`.
- [`examples/human_in_the_loop.py`](examples/human_in_the_loop.py): approval estático/dinámico,
  approve/reject, serialización y `Agent.resume()`.
- [`examples/openai_gateway.py`](examples/openai_gateway.py): endpoint compatible con OpenAI
  Responses.
- [`examples/list_openai_models.py`](examples/list_openai_models.py): consulta de modelos disponibles.

## Capacidades actuales

- Loop async acotado y multi-tool.
- Contratos de tools por type hints, Pydantic explícito o `ToolSpec` manual.
- Modos de resultado `continue`, `fallback` y `final`.
- Policy post-tool opcional.
- Human-in-the-loop con approval estático o dinámico, pause, approve/reject, serialización y resume.
- Historial y sesiones en memoria, Redis o PostgreSQL.
- Hooks locales y timeouts para providers/tools.
- OpenAI Responses directo o mediante un gateway compatible.

Google conserva un namespace para la evolución multiproveedor, pero todavía no incluye un engine
funcional. Anthropic y streaming permanecen fuera de este alfa.

## Sesiones

Las sesiones persistentes conservan el historial por `session_id` y lo ligan al provider/modelo
original. `MemorySessionStore`, Redis y PostgreSQL implementan el mismo contrato; Redis y PostgreSQL
permiten compartir estado entre procesos.

Durante una pausa por approval, Cortex conserva el turno activo y sólo lo termina cuando las calls
pendientes quedan resueltas. Una sesión dañada o deliberadamente descartada puede reiniciarse con
`await agent.reset_session(session_id)`.

## Licencia

Apache License 2.0.
