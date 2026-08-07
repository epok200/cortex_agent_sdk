import json
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Self, TypeIs

import pytest
from pydantic import BaseModel, Field, JsonValue, model_validator

from cortex_agent_sdk import final_answer, tool
from cortex_agent_sdk.engine import ToolCall
from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.immutable import thaw_json_object
from cortex_agent_sdk.tools.contracts import build_definition
from cortex_agent_sdk.tools.execution import ToolExecutor, ToolSet
from cortex_agent_sdk.tools.models import Injected, ToolBinding, ToolSpec


class ReminderArgs(BaseModel):
    message: str = Field(min_length=1, description="Texto del recordatorio.")
    when: datetime | None = None
    cron: str | None = Field(default=None, pattern=r"^\S+( \S+){4}$")

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if (self.when is None) == (self.cron is None):
            raise ValueError("se requiere exactamente when o cron")
        return self


@tool(input_model=ReminderArgs)
async def schedule_reminder(args: ReminderArgs) -> dict[str, str]:
    """Programa un recordatorio."""
    return {"message": args.message}


@final_answer(input_model=ReminderArgs)
async def confirm_reminder(args: ReminderArgs) -> str:
    """Confirma un recordatorio."""
    return args.message


def _is_json_object(value: JsonValue) -> TypeIs[dict[str, JsonValue]]:
    return isinstance(value, dict)


def test_decorators_preserve_callable_signatures() -> None:
    schedule: Callable[[ReminderArgs], Awaitable[dict[str, str]]] = schedule_reminder
    confirm: Callable[[ReminderArgs], Awaitable[str]] = confirm_reminder

    assert schedule is schedule_reminder
    assert confirm is confirm_reminder


def test_pydantic_model_builds_visible_contract() -> None:
    definition = build_definition(schedule_reminder)
    schema = thaw_json_object(definition.spec.parameters)
    properties = schema["properties"]

    assert _is_json_object(properties)
    message = properties["message"]
    assert _is_json_object(message)

    assert definition.input_parameter == "args"
    assert definition.spec.name == "schedule_reminder"
    assert definition.spec.description == "Programa un recordatorio."
    assert schema["additionalProperties"] is False
    assert set(properties) == {"message", "when", "cron"}
    assert message["minLength"] == 1
    assert message["description"] == "Texto del recordatorio."


def test_pydantic_contract_matches_manual_semantics() -> None:
    generated = thaw_json_object(build_definition(schedule_reminder).spec.parameters)
    manual = ToolSpec(
        name="schedule_reminder",
        description="Programa un recordatorio.",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Texto del recordatorio.",
                    "minLength": 1,
                },
                "when": {},
                "cron": {},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    )
    explicit = thaw_json_object(manual.parameters)

    generated_properties = generated["properties"]
    explicit_properties = explicit["properties"]
    assert _is_json_object(generated_properties)
    assert _is_json_object(explicit_properties)

    generated_message = generated_properties["message"]
    explicit_message = explicit_properties["message"]
    assert _is_json_object(generated_message)
    assert _is_json_object(explicit_message)

    assert generated["required"] == explicit["required"]
    assert generated["additionalProperties"] == explicit["additionalProperties"]
    assert generated_message["type"] == "string"
    assert generated_message["description"] == explicit_message["description"]
    assert generated_message["minLength"] == 1


@pytest.mark.asyncio
async def test_executor_passes_validated_model_instance() -> None:
    call = ToolCall(
        call_id="call-1",
        name="schedule_reminder",
        arguments={
            "message": "Pagar nómina",
            "when": "2026-08-10T09:00:00",
        },
    )

    outcome = await ToolExecutor(ToolSet.build((schedule_reminder,)), 1).execute(call)

    assert outcome.failed is False
    assert json.loads(outcome.output) == {"message": "Pagar nómina"}


@pytest.mark.asyncio
async def test_invalid_model_never_executes_tool() -> None:
    executions: list[ReminderArgs] = []

    @tool(input_model=ReminderArgs)
    async def guarded(args: ReminderArgs) -> str:
        executions.append(args)
        return "ok"

    call = ToolCall(
        call_id="call-2",
        name="guarded",
        arguments={"message": "Pagar nómina"},
    )
    outcome = await ToolExecutor(ToolSet.build((guarded,)), 1).execute(call)

    assert outcome.failed is True
    assert outcome.error_code == CodigoError.TOOL_ARGUMENTOS_INVALIDOS
    assert executions == []


@pytest.mark.asyncio
async def test_extra_arguments_are_rejected_for_declared_models() -> None:
    call = ToolCall(
        call_id="call-3",
        name="schedule_reminder",
        arguments={
            "message": "Pagar nómina",
            "when": "2026-08-10T09:00:00",
            "unexpected": True,
        },
    )

    outcome = await ToolExecutor(ToolSet.build((schedule_reminder,)), 1).execute(call)

    assert outcome.failed is True
    assert outcome.error_code == CodigoError.TOOL_ARGUMENTOS_INVALIDOS


@pytest.mark.asyncio
async def test_final_answer_supports_declared_models() -> None:
    call = ToolCall(
        call_id="call-4",
        name="confirm_reminder",
        arguments={
            "message": "Pagar nómina",
            "when": "2026-08-10T09:00:00",
        },
    )

    outcome = await ToolExecutor(ToolSet.build((confirm_reminder,)), 1).execute(call)

    assert outcome.failed is False
    assert outcome.final_answer is True
    assert outcome.output == "Pagar nómina"


def test_injected_arguments_stay_outside_pydantic_contract() -> None:
    class Context:
        pass

    context = Context()

    @tool(input_model=ReminderArgs)
    async def bound(
        args: ReminderArgs,
        context: Annotated[Context, Injected],
    ) -> str:
        return args.message

    definition = build_definition(
        ToolBinding(function=bound, private_arguments={"context": context})
    )
    schema = thaw_json_object(definition.spec.parameters)
    properties = schema["properties"]

    assert _is_json_object(properties)
    assert "context" not in properties
    assert set(definition.private_arguments) == {"context"}


def test_runtime_binding_can_declare_pydantic_model() -> None:
    async def dynamic(args: ReminderArgs) -> str:
        return args.message

    definition = build_definition(ToolBinding(function=dynamic, input_model=ReminderArgs))

    assert definition.input_model is ReminderArgs
    assert definition.input_parameter == "args"


def test_incompatible_callable_fails_during_registration() -> None:
    @tool(input_model=ReminderArgs)
    async def invalid(message: str) -> str:
        return message

    with pytest.raises(AppError) as caught:
        build_definition(invalid)

    assert caught.value.codigo == CodigoError.CONFIG_INVALIDA


def test_manual_toolspec_remains_available() -> None:
    @tool
    async def legacy(value: int) -> str:
        return str(value)

    spec = ToolSpec(
        name="legacy",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    definition = build_definition(ToolBinding(function=legacy, spec=spec))

    assert definition.spec is spec
    assert definition.input_parameter is None
