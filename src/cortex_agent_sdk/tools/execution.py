import asyncio
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from cortex_agent_sdk.engine import ToolCall
from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.history.models import ToolResultPart
from cortex_agent_sdk.immutable import thaw_json_object
from cortex_agent_sdk.tools.approval import ToolApprovalRule
from cortex_agent_sdk.tools.contracts import ToolDefinition, build_definition
from cortex_agent_sdk.tools.models import (
    ToolBinding,
    ToolFunction,
    ToolResultMode,
    ToolSpec,
)

_JSON_ADAPTER = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    call: ToolCall
    output: str
    failed: bool
    result_mode: ToolResultMode = ToolResultMode.CONTINUE
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call, ToolCall) or not isinstance(self.output, str):
            raise TypeError("ToolOutcome inválido")
        if not isinstance(self.failed, bool) or not isinstance(self.result_mode, ToolResultMode):
            raise TypeError("ToolOutcome inválido")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise TypeError("ToolOutcome inválido")

    @property
    def final_answer(self) -> bool:
        return self.result_mode is ToolResultMode.FINAL

    @property
    def fallback_answer(self) -> bool:
        return self.result_mode is ToolResultMode.FALLBACK

    @classmethod
    def skipped(cls, call: ToolCall) -> "ToolOutcome":
        return _failed_outcome(
            call,
            CodigoError.TOOL_OMITIDA_POR_LIMITE,
            "La herramienta se omitió porque se alcanzó el límite de fallas.",
        )

    @classmethod
    def rejected(cls, call: ToolCall, message: str) -> "ToolOutcome":
        payload = {"ok": False, "approval": "rejected", "message": message}
        output = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return cls(call=call, output=output, failed=False)

    def as_history_part(self) -> ToolResultPart:
        return ToolResultPart(
            call_id=self.call.call_id,
            name=self.call.name,
            output=self.output,
            is_error=self.failed,
            error_code=self.error_code,
        )


class ToolSet:
    def __init__(self, definitions: Mapping[str, ToolDefinition]) -> None:
        self._definitions = MappingProxyType(dict(definitions))

    @classmethod
    def build(cls, tools: Iterable[ToolFunction | ToolBinding]) -> "ToolSet":
        definitions: dict[str, ToolDefinition] = {}
        for item in tools:
            definition = build_definition(item)
            if definition.spec.name in definitions:
                raise AppError(
                    CodigoError.CONFIG_INVALIDA,
                    f"tool duplicada: {definition.spec.name}",
                )
            definitions[definition.spec.name] = definition
        return cls(definitions)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(definition.spec for definition in self._definitions.values())

    def definition(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def result_mode(self, name: str) -> ToolResultMode:
        definition = self._definitions.get(name)
        if definition is None:
            return ToolResultMode.CONTINUE
        return definition.result_mode

    def approval_rule(self, name: str) -> ToolApprovalRule | None:
        definition = self._definitions.get(name)
        if definition is None:
            return None
        return definition.needs_approval

    def has_approval_rule(self, name: str) -> bool:
        rule = self.approval_rule(name)
        return rule is not None and rule is not False

    def is_final_answer(self, name: str) -> bool:
        return self.result_mode(name) is ToolResultMode.FINAL


class ToolExecutor:
    def __init__(self, tools: ToolSet, timeout_seconds: float) -> None:
        self._tools = tools
        self._timeout_seconds = timeout_seconds

    def approval_arguments(self, call: ToolCall) -> Mapping[str, object] | None:
        """Devuelve argumentos públicos ya validados sin ejecutar la tool."""
        definition = self._tools.definition(call.name)
        if definition is None:
            return None
        validated = _validate_public_model(definition, call)
        if isinstance(validated, ToolOutcome):
            return None
        return MappingProxyType(validated.model_dump())

    async def execute(self, call: ToolCall) -> ToolOutcome:
        definition = self._tools.definition(call.name)
        if definition is None:
            return _failed_outcome(
                call,
                CodigoError.TOOL_NO_ENCONTRADA,
                "La herramienta solicitada no existe.",
            )

        validated = _validate_public_model(definition, call)
        if isinstance(validated, ToolOutcome):
            return validated
        arguments = _execution_arguments(definition, validated)

        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await definition.function(**arguments)
        except TimeoutError:
            return _failed_outcome(
                call,
                CodigoError.TOOL_TIMEOUT,
                "La herramienta agotó su tiempo de ejecución.",
                definition.result_mode,
            )
        except AppError as error:
            message = error.mensaje_seguro or "La herramienta no pudo completar la operación."
            return _failed_outcome(call, error.codigo, message, definition.result_mode)
        except Exception as error:
            detail = f"{call.name} lanzó {type(error).__name__}"
            raise AppError(
                CodigoError.TOOL_FALLO,
                detail,
                mensaje_seguro="La herramienta falló de forma inesperada.",
            ) from error

        if definition.is_final_answer and not isinstance(result, str):
            raise AppError(
                CodigoError.FINAL_ANSWER_INVALIDA,
                f"{call.name} debe devolver str",
            )
        output = _serialize_result(result, call.name)
        return ToolOutcome(call, output, False, definition.result_mode)


def _validate_public_model(
    definition: ToolDefinition,
    call: ToolCall,
) -> BaseModel | ToolOutcome:
    public_input = thaw_json_object(call.arguments)
    if definition.schema_validator is not None:
        schema_error = next(definition.schema_validator.iter_errors(public_input), None)
        if schema_error is not None:
            return _failed_outcome(
                call,
                CodigoError.TOOL_ARGUMENTOS_INVALIDOS,
                "Los argumentos de la herramienta son inválidos.",
                definition.result_mode,
            )
    try:
        return definition.input_model.model_validate(public_input, extra="forbid")
    except ValidationError:
        return _failed_outcome(
            call,
            CodigoError.TOOL_ARGUMENTOS_INVALIDOS,
            "Los argumentos de la herramienta son inválidos.",
            definition.result_mode,
        )


def _execution_arguments(
    definition: ToolDefinition,
    validated: BaseModel,
) -> dict[str, object]:
    if definition.input_parameter is not None:
        public_arguments: dict[str, object] = {definition.input_parameter: validated}
    else:
        public_arguments = validated.model_dump()
    return {**public_arguments, **definition.private_arguments}


def _serialize_result(result: object, tool_name: str) -> str:
    if isinstance(result, str):
        return result
    try:
        value = _JSON_ADAPTER.validate_python(result)
    except ValidationError as error:
        raise AppError(
            CodigoError.TOOL_RESULTADO_INVALIDO,
            f"{tool_name} devolvió un resultado no serializable",
        ) from error
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _failed_outcome(
    call: ToolCall,
    code: CodigoError,
    message: str,
    result_mode: ToolResultMode = ToolResultMode.CONTINUE,
) -> ToolOutcome:
    payload = {"ok": False, "error": {"code": code.value, "message": message}}
    output = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return ToolOutcome(call, output, True, result_mode, code.value)
