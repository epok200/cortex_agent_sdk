import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, JsonValue, create_model

from cortex_agent_sdk.errores import AppError, CodigoError
from cortex_agent_sdk.immutable import thaw_json_object
from cortex_agent_sdk.tools.models import Injected, ToolBinding, ToolFunction, ToolSpec


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    spec: ToolSpec
    function: ToolFunction
    input_model: type[BaseModel]
    private_arguments: Mapping[str, object]
    is_final_answer: bool
    schema_validator: Draft202012Validator | None


def build_definition(item: ToolFunction | ToolBinding) -> ToolDefinition:
    if isinstance(item, ToolBinding):
        binding = item
    else:
        binding = ToolBinding(function=item)
        if not getattr(item, "__cortex_tool__", False):
            raise AppError(
                CodigoError.CONFIG_INVALIDA,
                f"{item.__name__} requiere @tool o ToolBinding",
            )

    function = inspect.unwrap(binding.function)
    if not inspect.iscoroutinefunction(function):
        raise AppError(CodigoError.CONFIG_INVALIDA, f"{function.__name__} debe ser async")

    input_model, inferred_spec = _infer_contract(function, binding.private_arguments)
    spec = binding.spec or inferred_spec
    schema_validator = _explicit_validator(binding.spec, inferred_spec)
    private_arguments = MappingProxyType(dict(binding.private_arguments))
    is_final = bool(getattr(function, "__cortex_final_answer__", False))
    return ToolDefinition(
        spec,
        binding.function,
        input_model,
        private_arguments,
        is_final,
        schema_validator,
    )


def _infer_contract(
    function: ToolFunction,
    private_arguments: Mapping[str, object],
) -> tuple[type[BaseModel], ToolSpec]:
    signature = inspect.signature(function)
    try:
        annotations = get_type_hints(function, include_extras=True)
    except (NameError, TypeError) as error:
        raise AppError(
            CodigoError.CONFIG_INVALIDA,
            f"no se resolvieron tipos de {function.__name__}",
        ) from error

    fields: dict[str, Any] = {}
    injected_names: set[str] = set()
    for name, parameter in signature.parameters.items():
        annotation = annotations.get(name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            raise AppError(CodigoError.CONFIG_INVALIDA, f"falta tipo en {function.__name__}.{name}")
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            raise AppError(CodigoError.CONFIG_INVALIDA, f"{function.__name__} no admite variádicos")
        if _is_injected(annotation):
            injected_names.add(name)
            continue
        default = parameter.default
        if default is inspect.Signature.empty:
            default = ...
        fields[name] = (annotation, default)

    if injected_names != set(private_arguments):
        raise AppError(
            CodigoError.CONFIG_INVALIDA,
            f"inyección inválida en {function.__name__}",
        )

    config = ConfigDict(extra="forbid")
    model_name = f"{function.__name__.title().replace('_', '')}Arguments"
    input_model = create_model(model_name, __config__=config, **fields)
    schema = input_model.model_json_schema()
    schema.pop("title", None)
    description = (inspect.getdoc(function) or "").split("\n", maxsplit=1)[0]
    spec = ToolSpec(
        name=function.__name__,
        description=description,
        parameters=schema,
    )
    return input_model, spec


def _is_injected(annotation: object) -> bool:
    if get_origin(annotation) is not Annotated:
        return False
    return Injected in get_args(annotation)[1:]


def _explicit_validator(
    explicit: ToolSpec | None,
    inferred: ToolSpec,
) -> Draft202012Validator | None:
    if explicit is None:
        return None
    schema = _plain_schema(explicit)
    inferred_schema = _plain_schema(inferred)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise AppError(
            CodigoError.CONFIG_INVALIDA,
            f"schema inválido en {explicit.name}",
        ) from error

    properties = schema.get("properties")
    inferred_properties = inferred_schema.get("properties")
    required = schema.get("required", [])
    inferred_required = inferred_schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(inferred_properties, dict):
        raise AppError(CodigoError.CONFIG_INVALIDA, f"properties inválido en {explicit.name}")
    explicit_fields = set(properties)
    inferred_fields = set(inferred_properties)
    if not explicit_fields <= inferred_fields:
        raise AppError(CodigoError.CONFIG_INVALIDA, f"campos incompatibles en {explicit.name}")
    if not isinstance(required, list) or not isinstance(inferred_required, list):
        raise AppError(CodigoError.CONFIG_INVALIDA, f"required inválido en {explicit.name}")
    explicit_required = set(required)
    callable_required = set(inferred_required)
    valid_required = explicit_required <= explicit_fields and callable_required <= explicit_required
    if not valid_required:
        raise AppError(CodigoError.CONFIG_INVALIDA, f"required incompatible en {explicit.name}")
    return Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def _plain_schema(spec: ToolSpec) -> dict[str, JsonValue]:
    return thaw_json_object(spec.parameters)
