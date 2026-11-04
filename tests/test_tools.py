"""Tools: schemas from signatures and from runtime schemas, and how their
handlers are called."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal, Optional

import pytest
from fakeserver import check_balance

import ulai_agents as agent


def spec(t: agent.Tool):
    return t._spec()


def test_schema_from_the_signature():
    s = spec(check_balance)
    assert (s.name, s.description) == ("check_balance", "Look up a balance.")
    acct, kind, limit = s.parameters["account"], s.parameters["kind"], s.parameters["limit"]
    assert (acct.type, acct.description, list(acct.enum)) == ("string", "The account number.", [])
    assert (kind.type, kind.description, list(kind.enum)) == ("string", "Which balance.", ["savings", "current"])
    assert (limit.type, limit.description) == ("integer", "")
    assert list(s.required) == ["account"]  # kind and limit are optional
    assert check_balance._problems() == []


def test_every_supported_type():
    @agent.tool(description="d")
    def t(
        s: str,
        b: bool,
        i: int,
        f: float,
        lit: Literal["a", "b"],
        opt: Optional[int],  # noqa: UP045  (the old spelling works too)
        union_none: str | None,
        ann_opt: Annotated[float | None, "maybe"],
        opt_ann: Annotated[bool, "flag"] | None,
        lit_none: Literal["x", None],
        default: str = "d",
    ) -> None: ...

    s = spec(t)
    kinds = {k: (v.type, list(v.enum), v.description) for k, v in s.parameters.items()}
    assert kinds == {
        "s": ("string", [], ""),
        "b": ("boolean", [], ""),
        "i": ("integer", [], ""),
        "f": ("number", [], ""),
        "lit": ("string", ["a", "b"], ""),
        "opt": ("integer", [], ""),
        "union_none": ("string", [], ""),
        "ann_opt": ("number", [], "maybe"),
        "opt_ann": ("boolean", [], "flag"),
        "lit_none": ("string", ["x"], ""),
        "default": ("string", [], ""),
    }
    assert list(s.required) == ["s", "b", "i", "f", "lit"]
    assert t._problems() == []


def test_string_annotations_are_resolved():
    def f(account: "Annotated[str, 'The account.']", n: "int | None" = None) -> None: ...

    s = spec(agent.tool(f, description="d"))
    assert s.parameters["account"].description == "The account." and s.parameters["n"].type == "integer"
    assert list(s.required) == ["account"]


def test_name_and_description_default_to_the_function():
    @agent.tool
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return "shipped"

    assert (lookup_order.name, lookup_order.description) == ("lookup_order", "Look up an order.")
    assert lookup_order("o-1") == "shipped"  # still callable as the function

    @agent.tool(name="find", description="Find it.")
    def g() -> None: ...

    assert (g.name, g.description) == ("find", "Find it.")

    @agent.tool
    def undocumented() -> None: ...

    assert "needs a description" in undocumented._problems()[0]


@pytest.mark.parametrize(
    "fn, want",
    [
        (lambda x: None, "argument x needs a type annotation"),
        (lambda *args: None, "*args cannot be described"),
        (lambda **kw: None, "*kw cannot be described"),
    ],
)
def test_signatures_that_cannot_be_described(fn, want):
    assert want in agent.tool(fn, name="t", description="d")._problems()[0]


def test_unsupported_types_are_validation_errors():
    def lists(x: list[str]) -> None: ...
    def dicts(x: dict) -> None: ...
    def unions(x: str | int) -> None: ...
    def literal_ints(x: Literal[1, 2]) -> None: ...
    def objects(x: object) -> None: ...
    def positional(x: str, /) -> None: ...
    def unresolved(x: "Nope") -> None: ...  # noqa: F821

    for fn, want in [
        (lists, "argument x is list[str]; use str, bool, int, float or a Literal of strings"),
        (dicts, "argument x is dict; use str"),
        (unions, "argument x is a union of"),
        (literal_ints, "a Literal of anything but strings"),
        (objects, "argument x is object"),
        (positional, "positional-only"),
        (unresolved, "cannot be resolved"),
    ]:
        t = agent.tool(fn, description="d")
        assert t._problems() and want in t._problems()[0], (fn.__name__, t._problems())
        a = agent.Agent(id="a", instructions="i", cloud=agent.CloudAccount("p", credentials_json="{}"), tools=[t])
        with pytest.raises(agent.ValidationError, match="use str|union|Literal|positional|resolved"):
            a.validate()


def test_tool_from_schema():
    got: dict = {}

    def handler(args: dict) -> str:
        got.update(args)
        return "shipped"

    t = agent.tool_from_schema(
        "lookup_order",
        "Look up an order.",
        {
            "order_id": agent.Param(type=agent.ParamType.STRING, description="The order."),
            "express": agent.Param(type=agent.ParamType.BOOLEAN),
            "tier": agent.Param(type="string", enum=["gold", "silver"]),
            "qty": agent.Param(type=agent.ParamType.INTEGER),
            "weight": agent.Param(type="number"),
        },
        ["order_id"],
        handler,
    )
    assert t._problems() == []
    s = spec(t)
    assert s.parameters["express"].type == "boolean" and list(s.parameters["tier"].enum) == ["gold", "silver"]
    assert s.parameters["order_id"].description == "The order."
    assert (s.parameters["qty"].type, s.parameters["weight"].type) == ("integer", "number")
    assert list(s.required) == ["order_id"]

    out = asyncio.run(t._invoke('{"order_id":"o-1","express":true,"extra":[1]}'))
    assert out == "shipped" and got == {"order_id": "o-1", "express": True, "extra": [1]}

    # Declared only: no handler, so this client does not claim it.
    assert agent.tool_from_schema("end_call", "Ends the call.").handler is None


@pytest.mark.parametrize(
    "params, required, want",
    [
        ({"x": agent.Param(type="array")}, [], 'parameter x has type "array"'),
        ({"x": agent.Param()}, ["y"], "required parameter y is not among its parameters"),
        ({"x": {"type": "string"}}, [], "parameter x must be an agent.Param"),
    ],
)
def test_tool_from_schema_problems(params, required, want):
    t = agent.tool_from_schema("t", "d", params, required)
    assert want in t._problems()[0]


async def test_arguments_are_decoded_for_the_handler():
    seen = []

    @agent.tool(description="d")
    def t(name: str, n: int, ratio: float, flag: bool, kind: str | None, limit: int = 5) -> dict:
        seen.append((name, n, ratio, flag, kind, limit))
        return {"ok": 1}

    assert await t._invoke('{"name":"a","n":2.0,"ratio":1,"flag":false,"unknown":"x"}') == {"ok": 1}
    assert seen[-1] == ("a", 2, 1.0, False, None, 5)
    assert type(seen[-1][1]) is int and type(seen[-1][2]) is float
    await t._invoke('{"name":"a","n":2,"ratio":0.5,"flag":true,"kind":null,"limit":null}')
    assert seen[-1] == ("a", 2, 0.5, True, None, 5)

    for raw, want in [
        ('{"n":2,"ratio":1,"flag":true}', "name is required"),
        ('{"name":null,"n":2,"ratio":1,"flag":true}', "name is required"),
        ('{"name":1,"n":2,"ratio":1,"flag":true}', "name must be a string"),
        ('{"name":"a","n":2.5,"ratio":1,"flag":true}', "n must be a integer"),
        ('{"name":"a","n":true,"ratio":1,"flag":true}', "n must be a integer"),
        ('{"name":"a","n":1,"ratio":"x","flag":true}', "ratio must be a number"),
        ('{"name":"a","n":1,"ratio":1,"flag":"yes"}', "flag must be a boolean"),
        ("{not json", "were not valid"),
        ("[1, 2]", "expected a JSON object"),
    ]:
        with pytest.raises(agent.ToolError, match=want):
            await t._invoke(raw)


async def test_no_arguments():
    @agent.tool(description="Ping.")
    async def ping() -> str:
        return "pong"

    assert await ping._invoke("") == "pong"
    assert await ping._invoke("{}") == "pong"
    assert spec(ping).parameters == {} and list(spec(ping).required) == []
