"""Tools: what an agent can do, as functions the model may call."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import re
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from ulai.agent.v1 import agent_pb2 as pb

from ._models import ParamType, values, wire

__all__ = ["Param", "Tool", "ToolError", "tool", "tool_from_schema"]

# What the models accept as a function name. fullmatch, so a trailing newline
# cannot slip through the way it would past a bare "$".
_TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")

_KINDS: dict[Any, str] = {str: "string", bool: "boolean", int: "integer", float: "number"}


class ToolError(Exception):
    """Raise from a tool handler to tell the model what went wrong.

    The model is answered ``{"error": "<your message>"}``, so it can apologise
    or try something else. Any other exception answers
    ``{"error": "the tool failed"}`` and is logged.
    """


@dataclass(frozen=True)
class Param:
    """One parameter of a tool built with ``tool_from_schema``."""

    type: ParamType | str = ParamType.STRING
    description: str = ""
    enum: Sequence[str] | None = None
    """When set, limits the value to these."""


class Tool:
    """Something the agent can do: a function the model may call.

    Make one with the ``@agent.tool`` decorator, or ``agent.tool_from_schema``
    for parameters only known at run time. A Tool is still callable as the
    function it wraps.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Mapping[str, Param],
        required: Sequence[str],
        handler: Callable[..., Any] | None,
        *,
        problem: str | None = None,
        signature: bool = False,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = dict(parameters)
        self._required = list(required)
        self._handler = handler
        self._problem = problem
        # Whether the handler takes the arguments as keywords (from its
        # signature) rather than as one dict (tool_from_schema).
        self._signature = signature
        if handler is not None and signature:
            functools.update_wrapper(self, handler, updated=())

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Param]:
        return dict(self._parameters)

    @property
    def required(self) -> list[str]:
        return list(self._required)

    @property
    def handler(self) -> Callable[..., Any] | None:
        """The function that answers the tool, or None for a tool this client
        declares but does not answer."""
        return self._handler

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._handler is None:
            raise TypeError(f"tool {self._name} has no handler")
        return self._handler(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Tool(name={self._name!r}, description={self._description!r}, "
            f"parameters={sorted(self._parameters)!r}, required={self._required!r})"
        )

    def _problems(self) -> list[str]:
        if self._problem:
            return [self._problem]
        if not isinstance(self._name, str) or not _TOOL_NAME.fullmatch(self._name):
            return [
                f"tool name {self._name!r} must start with a letter or underscore and use "
                "only letters, digits, _ . -, at most 64 characters"
            ]
        if not isinstance(self._description, str) or not self._description.strip():
            return [f'tool "{self._name}" needs a description: it is how the model knows when to call it']
        return []

    def _spec(self) -> pb.ToolSpec:
        spec = pb.ToolSpec(name=self._name, description=self._description, required=self._required)
        for pname, p in self._parameters.items():
            spec.parameters[pname].CopyFrom(
                pb.ToolParam(type=wire(p.type), description=p.description, enum=list(p.enum or ()))
            )
        return spec

    async def _invoke(self, args_json: str) -> Any:
        """Run the handler on the model's arguments. Raises ToolError for
        arguments it cannot take."""
        assert self._handler is not None
        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError as e:
            raise ToolError(f"the arguments for {self._name} were not valid: {e}") from None
        if not isinstance(args, dict):
            raise ToolError(f"the arguments for {self._name} were not valid: expected a JSON object")
        if not self._signature:
            return await _call(self._handler, args)
        return await _call(self._handler, **self._decode(args))

    def _decode(self, args: dict[str, Any]) -> dict[str, Any]:
        """The model's arguments as the handler's keyword arguments.

        Unknown names are dropped, as Go's JSON decoding drops unknown fields.
        A missing or null optional argument takes the handler's default, or
        None when the argument has none.
        """
        sig = inspect.signature(self._handler)  # type: ignore[arg-type]
        out: dict[str, Any] = {}
        for pname, p in self._parameters.items():
            value = args.get(pname)
            if value is None:
                if pname in self._required:
                    raise ToolError(f"the arguments for {self._name} were not valid: {pname} is required")
                if sig.parameters[pname].default is inspect.Parameter.empty:
                    out[pname] = None
                continue
            out[pname] = _coerce(self._name, pname, wire(p.type), value)
        return out


def _coerce(tool_name: str, pname: str, kind: str, value: Any) -> Any:
    bad = ToolError(f"the arguments for {tool_name} were not valid: {pname} must be a {kind}, not {value!r}")
    if kind == "string":
        if not isinstance(value, str):
            raise bad
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            raise bad
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise bad
    if kind == "integer":
        if isinstance(value, float):
            if not value.is_integer():
                raise bad
            return int(value)
        return value
    return float(value)


async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a sync or async handler. A sync one runs in a thread, so a slow
    lookup does not hold up the call's events."""
    if _is_async(fn):
        return await fn(*args, **kwargs)
    result = await asyncio.to_thread(fn, *args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _is_async(fn: Any) -> bool:
    # A callable object counts when its __call__ is async.
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(type(fn).__call__)


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Declare a tool from a function. Its parameters come from the signature::

        @agent.tool(description="Look up the balance of an account.")
        async def check_balance(
            account: Annotated[str, "The account number."],
            kind: Literal["savings", "current"] | None = None,
        ) -> dict:
            return {"balance": 250}

    Parameter types are ``str``, ``bool``, ``int`` and ``float``; a ``Literal``
    of strings limits the value to those. A parameter is optional when it has
    a default or is ``X | None``. ``Annotated[T, "text"]`` describes it to the
    model. The name defaults to the function's, the description to its
    docstring.

    The handler may be sync or async. What it returns is sent to the model as
    JSON (None sends ``{"ok": true}``); raise ``agent.ToolError`` to report a
    failure to the model.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        tname = name if name is not None else getattr(f, "__name__", "")
        desc = description if description is not None else (inspect.getdoc(f) or "")
        params, required, problem = _schema_of(tname, f)
        return Tool(tname, desc, params, required, f, problem=problem, signature=True)

    if fn is None:
        return wrap
    return wrap(fn)


def tool_from_schema(
    name: str,
    description: str,
    params: Mapping[str, Param] | None = None,
    required: Sequence[str] | None = None,
    handler: Callable[[dict[str, Any]], Any] | None = None,
) -> Tool:
    """Declare a tool whose parameters are only known at run time — read from
    a database, say — rather than from a function's signature.

    The handler gets the model's arguments as one dict. A None handler
    declares the tool without answering it here, for a tool another client
    answers.
    """
    params = dict(params or {})
    required = list(required or [])
    problem = None
    for pname, p in params.items():
        if not isinstance(p, Param):
            problem = problem or f"tool {name}: parameter {pname} must be an agent.Param, not {type(p).__name__}"
        elif not isinstance(p.type, str) or wire(p.type) not in values(ParamType):
            problem = problem or (
                f'tool {name}: parameter {pname} has type "{wire(p.type)}"; use agent.ParamType.STRING, '
                "BOOLEAN, INTEGER or NUMBER"
            )
    for r in required:
        if r not in params:
            problem = problem or f"tool {name}: required parameter {r} is not among its parameters"
    return Tool(name, description, params, required, handler, problem=problem, signature=False)


def _schema_of(name: str, fn: Callable[..., Any]) -> tuple[dict[str, Param], list[str], str | None]:
    """A tool's parameters, read off its handler's signature."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        return {}, [], f"tool {name}: cannot read the handler's signature: {e}"
    try:
        hints = _hints(fn)
    except Exception as e:
        return {}, [], f"tool {name}: an argument's type annotation cannot be resolved: {e}"
    params: dict[str, Param] = {}
    required: list[str] = []
    for p in sig.parameters.values():
        p = p.replace(annotation=hints.get(p.name, p.annotation))
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return {}, [], f"tool {name}: *{p.name} cannot be described to the model; name each argument"
        if p.kind is p.POSITIONAL_ONLY:
            return {}, [], f"tool {name}: argument {p.name} is positional-only; arguments are passed by name"
        if p.annotation is p.empty:
            return {}, [], (
                f"tool {name}: argument {p.name} needs a type annotation: "
                "use str, bool, int, float or a Literal of strings"
            )
        described = _describe(p.annotation)
        if isinstance(described, str):
            return {}, [], (
                f"tool {name}: argument {p.name} is {described}; use str, bool, int, float or a Literal of strings"
            )
        kind, enum, desc, optional = described
        params[p.name] = Param(type=ParamType(kind), description=desc, enum=enum)
        if not optional and p.default is p.empty:
            required.append(p.name)
    return params, required, None


def _hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """The handler's annotations, with string ones (and from __future__ import
    annotations) resolved."""
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except TypeError:
        # Not a plain function (a partial, a callable object): read the
        # signature, which knows how to unwrap them.
        sig = inspect.signature(fn, eval_str=True)
        return {k: v.annotation for k, v in sig.parameters.items() if v.annotation is not v.empty}


def _describe(ann: Any) -> tuple[str, list[str] | None, str, bool] | str:
    """(kind, enum, description, optional) for an annotation, or what it is
    when it cannot be described."""
    desc = ""
    optional = False
    while True:
        origin = get_origin(ann)
        if origin is Annotated:
            base, *meta = get_args(ann)
            for m in meta:
                if isinstance(m, str) and not desc:
                    desc = m
            ann = base
            continue
        if origin is Union or origin is types.UnionType:
            members = get_args(ann)
            rest = [a for a in members if a is not type(None)]
            if len(rest) != len(members):
                optional = True
            if len(rest) == 1:
                ann = rest[0]
                continue
            return f"a union of {_type_name(ann)}"
        break
    if get_origin(ann) is Literal:
        choices = [v for v in get_args(ann) if v is not None]
        if len(choices) != len(get_args(ann)):
            optional = True
        if choices and all(isinstance(v, str) for v in choices):
            return "string", choices, desc, optional
        return f"{_type_name(ann)} (a Literal of anything but strings)"
    kind = _KINDS.get(ann)
    if kind is None:
        return _type_name(ann)
    return kind, None, desc, optional


def _type_name(ann: Any) -> str:
    if isinstance(ann, type) and get_origin(ann) is None:
        return ann.__name__
    return repr(ann).replace("typing.", "")
