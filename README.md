# Cortex Agent SDK

SDK async para construir agentes con una API pequeña y control explícito del loop, las tools, el
historial y el ciclo de vida.

> Cortex Agent SDK está en alfa. La API puede cambiar antes de la versión `1.0.0`.

## Requisitos

- Python `>=3.13`.
- Una credencial del provider elegido.

## Instalación

Para OpenAI Responses:

```bash
uv add "cortex-agent-sdk[openai]"
```

También puede instalarse con `pip`:

```bash
python -m pip install "cortex-agent-sdk[openai]"
```

El core instala únicamente Pydantic y JSON Schema. Los providers y transportes se habilitan mediante
extras opcionales:

- `openai`: engine de OpenAI Responses.
- `gateway`: conexión mediante un endpoint compatible con OpenAI Responses.
- `redis`: sesiones persistentes en Redis.
- `postgres`: sesiones persistentes en PostgreSQL.
- `all`: todas las integraciones disponibles.

## Uso mínimo

El SDK oficial de OpenAI lee `OPENAI_API_KEY` del entorno.

```python
import asyncio

from cortex_agent_sdk import Agent
from cortex_agent_sdk.openai import OpenAIEngine, OpenAIOptions


async def main() -> None:
    options = OpenAIOptions(max_output_tokens=128, reasoning_effort="low")
    async with Agent(OpenAIEngine("gpt-5.6-luna", options=options)) as agent:
        result = await agent.run("Responde únicamente: hola")
    print(result.text)


asyncio.run(main())
```

## Gateway compatible

`OpenAIEngine` también acepta un endpoint que conserve el protocolo de OpenAI Responses:

```python
import os

from cortex_agent_sdk.gateway import OpenAICompatibleGateway
from cortex_agent_sdk.openai import OpenAIEngine


gateway = OpenAICompatibleGateway(
    url=os.environ["CORTEX_GATEWAY_URL"],
    api_key=os.environ["CORTEX_GATEWAY_API_KEY"],
)
engine = OpenAIEngine("your-model", gateway=gateway)
```

## Tools

Una función async decorada puede exponerse al modelo como respuesta final:

```python
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
        OpenAIEngine("gpt-5.6-luna"),
        instructions=instructions,
        tools=(sumar,),
    ) as agent:
        result = await agent.run("Suma 20 y 22.")
    print(result.text)


asyncio.run(main())
```

Los type hints son la vía recomendada para tools sencillas. Cortex infiere el contrato visible y
valida los argumentos antes de ejecutar la función; no hace falta escribir JSON Schema ni
`ToolSpec` manualmente.

### Contratos Pydantic

Cuando los argumentos forman un contrato reutilizable o tienen validaciones entre campos, puede
usarse un `BaseModel` como fuente de verdad:

```python
from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from cortex_agent_sdk import final_answer


class ReminderArgs(BaseModel):
    message: str = Field(min_length=1, description="Texto del recordatorio.")
    when: datetime | None = None
    cron: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if (self.when is None) == (self.cron is None):
            raise ValueError("se requiere exactamente when o cron")
        return self


@final_answer(input_model=ReminderArgs)
async def schedule_reminder(args: ReminderArgs) -> str:
    """Programa un recordatorio."""
    return f"Recordatorio: {args.message}"
```

Cortex genera el `ToolSpec` neutral a partir del modelo, rechaza propiedades adicionales y entrega a
la función una instancia ya validada. `ToolBinding(input_model=...)` ofrece la misma capacidad para
tools construidas en runtime. El `ToolSpec` manual continúa disponible como escape hatch cuando el
schema visible necesita construirse dinámicamente o requiere control de bajo nivel.

### Control del resultado de tools

Cortex `0.1` separa el contrato de argumentos de la semántica del resultado:

- `@tool`: ejecuta y devuelve el resultado al modelo para continuar el loop.
- `@fallback_answer`: continúa el loop, pero conserva el último resultado exitoso como respaldo si el
  modelo termina limpiamente sin texto visible.
- `@final_answer`: hard stop; el resultado exitoso termina el run sin volver al modelo.

Un fallback permite reutilizar la misma tool de lectura tanto en una consulta simple como dentro de
un workflow compuesto:

```python
from cortex_agent_sdk import fallback_answer


@fallback_answer
async def consultar_agenda(fecha: str) -> str:
    return f"Agenda para {fecha}: ..."
```

Si el modelo llama `consultar_agenda`, después puede seguir usando otras tools. El fallback solo se
usa cuando la terminación posterior es limpia, no hay texto visible y no existe `response_model`.
Nunca sustituye `MAX_STEPS`, límite de fallos ni un stop incompleto del provider.

### Policy post-tool

Cuando terminar depende del contexto del run y no de una tool concreta, `Agent` acepta una policy
opcional. La policy recibe los outcomes de la última tanda junto con historial, paso y `session_id` y
decide entre continuar o terminar explícitamente:

```python
from cortex_agent_sdk import CONTINUE, Agent, FinalOutput


def policy(context):
    if any(outcome.output == "objetivo alcanzado" for outcome in context.outcomes):
        return FinalOutput("Listo.")
    return CONTINUE


agent = Agent(engine, tools=tools, tool_result_policy=policy)
```

La policy puede ser sync o async. `@final_answer` conserva prioridad como hard stop y no es
sustituido por la policy.

### Human-in-the-loop

Las acciones sensibles pueden requerir aprobación humana antes de ejecutarse:

```python
from cortex_agent_sdk import Agent, PendingRun, tool


@tool(needs_approval=True)
async def eliminar_evento(event_id: str) -> str:
    return f"Eliminado {event_id}"


result = await agent.run("Elimina el evento de mañana")
if result.pending_run is not None:
    pending = result.pending_run
    pending.approve(result.interruptions[0].call_id)
    result = await agent.resume(pending)
```

Mientras existe una interruption, la tool sensible todavía no se ha ejecutado. Un rechazo tampoco la
ejecuta: Cortex produce un resultado de rechazo visible para el modelo y continúa el loop sin pasar
esa call por los hooks ni por el executor. La decisión es por `call_id`; si el modelo intenta de nuevo
la misma acción genera una nueva call y vuelve a evaluarse la regla de aprobación.

La call aprobada queda ligada a su nombre y argumentos originales. Al reanudar, un `before_tool` no
puede modificar una call ya aprobada: lo que el usuario autorizó es exactamente lo que Cortex permite
ejecutar.

`needs_approval` también puede ser un callable sync o async. Cortex valida primero los argumentos
públicos de la tool y entrega al predicate el contexto del run, los argumentos parseados y el
`call_id`:

```python
from collections.abc import Mapping

from cortex_agent_sdk import ToolApprovalContext, tool


def requiere_revision(
    context: ToolApprovalContext,
    argumentos: Mapping[str, object],
    call_id: str,
) -> bool:
    return argumentos["cantidad"] > 10


@tool(needs_approval=requiere_revision)
async def eliminar_eventos(cantidad: int) -> str:
    return f"Eliminados {cantidad} eventos"
```

Si los argumentos son inválidos, el predicate no se ejecuta y la llamada sigue el flujo normal de
error de argumentos. El callable permite que una misma tool sea automática para operaciones seguras y
requiera intervención humana sólo cuando los datos concretos de esa call lo ameritan.

`PendingRun` es serializable mediante `to_json()` y recuperable con `PendingRun.from_json(...)`. Si el
run usó tools entregadas directamente a `Agent.run(...)`, deben proporcionarse de nuevo a
`Agent.resume(...)`. Si utilizó `response_model`, debe entregarse el mismo modelo al reanudar. Con
sesiones persistentes, Cortex conserva el `active_turn` durante la pausa y lo cierra únicamente al
resolver la tanda pendiente.

## Capacidades del alfa

- Loop async acotado y multi-tool.
- Tools async con schema inferido, Pydantic explícito o `ToolSpec` manual.
- Semánticas `continue`, `fallback` y `final` para resultados de tools.
- Policy post-tool opcional.
- Human-in-the-loop con approval estático o dinámico, pause, approve/reject, serialización y resume.
- Historial y sesiones en memoria, Redis o PostgreSQL.
- Hooks locales.
- Timeouts para providers y tools.
- Respuesta tipada con texto, usage, razón de salida y respuesta raw.
- OpenAI Responses directo o mediante un gateway compatible.

Google conserva un namespace estable para la evolución multiproveedor, pero todavía no incluye un
engine funcional. Anthropic y streaming están fuera de este alfa.

## Sesiones persistentes

Redis y PostgreSQL implementan el mismo contrato de sesiones que el store en memoria. El historial
queda aislado por `session_id`, ligado al provider y modelo originales, y protegido con lease
renovable, fencing token y compare-and-swap.

Redis no requiere inicialización de schema:

```python
import asyncio
import os

from cortex_agent_sdk import Agent
from cortex_agent_sdk.openai import OpenAIEngine
from cortex_agent_sdk.redis import RedisSessionStore


async def main() -> None:
    store = RedisSessionStore(os.environ["REDIS_URL"])
    async with Agent(
        OpenAIEngine("gpt-5.6-luna"),
        session_store=store,
        own_session_store=True,
    ) as agent:
        result = await agent.run("Hola", session_id="producto:tenant:usuario")
    print(result.text)


asyncio.run(main())
```

PostgreSQL exige crear su tabla de forma explícita una vez:

```python
import asyncio
import os

from cortex_agent_sdk.postgres import PostgresSessionStore


async def main() -> None:
    store = PostgresSessionStore(os.environ["POSTGRES_URL"])
    try:
        await store.setup()
    finally:
        await store.aclose()


asyncio.run(main())
```

Una tarea periódica puede ejecutar `await store.cleanup_expired()` para vaciar historiales vencidos
que nunca volvieron a solicitarse. El row mínimo permanece para conservar el fencing counter.

`Agent.aclose()` hace un cierre ordenado: deja de aceptar turnos nuevos, espera los turnos activos y
después cierra los recursos que posee. El límite se configura con
`AgentOptions.shutdown_timeout_seconds` y cubre tanto el drenado como el cierre físico. Si vence
mientras hay un turno activo, el SDK no lo cancela ni empieza a cerrar recursos. Si vence durante el
cierre físico, algunos recursos podrían haberse cerrado ya. El timeout es un presupuesto de cierre,
no una garantía estricta de tiempo de pared: un finalizador que resista la cancelación puede retrasar
el retorno para no abandonar recursos a medias. Al excederlo regresa `RUNTIME_CIERRE_TIMEOUT` y una
segunda llamada a `aclose()` reintenta lo pendiente.

Una sesión dañada o deliberadamente descartada se elimina mediante
`await agent.reset_session(session_id)`.

## Licencia

Apache License 2.0.
