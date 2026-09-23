"""The node types an FFSL source file parses into."""

from __future__ import annotations

__all__ = [
    "AstNode",
    "Bool",
    "Int",
    "Float",
    "Str",
    "UnaryExpr",
    "BinaryExpr",
    "Ref",
    "FuncCall",
    "TypeRef",
    "PassStmt",
    "AssignmentStmt",
    "UpdateStmt",
    "ReturnStmt",
    "IfStmt",
    "ElifSection",
    "ElseSection",
    "Parameter",
    "Func",
    "Source",
    "LocalVariable",
    "Block",
    "Option",
    "Config",
    "TickVar",
    "TickData",
    "TileVar",
    "TileData",
    "PoissonDist",
    "NormalDist",
    "DeterministicDist",
    "CreateEmbers",
    "EmberJumpLikelihood",
    "EmberDeathProb",
    "EmberIgnitionProb",
    "CreateFlames",
    "FlameSpreadWeight",
    "FlameIgnitionProb",
    "BurnTime",
    "FireModel",
]

from collections import ChainMap
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .error import *


class AstNode(BaseModel):
    """Base of every node, which carries its source position and its children.

    `children` holds every node that the generic tree walks must visit.
    It never holds a node that sits up the tree or sideways in a scope.
    """

    pos: Position = Field(repr=False)
    # Why the edges up and sideways stay out of `children`
    # is in the developer notes, under the three kinds of edge.
    children: list[AstNode] = Field(repr=False)

    # Pydantic has no schema for `ChainMap`,
    # which the `scope` fields of the subclasses hold.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Bool(AstNode):
    """A `True` or `False` literal."""

    value: bool


class Int(AstNode):
    """An integer literal."""

    value: int


class Float(AstNode):
    """A floating point literal."""

    value: float


class Str(AstNode):
    """A string literal, with the quotes already stripped."""

    value: str


Literal = Bool | Int | Float | Str


class Ref(AstNode):
    """A name, or a dotted `object.attribute` pair.

    `scope` is attached after parsing, and the lookup properties need it.
    """

    name: str | tuple[str, str]
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)

    def __str__(self) -> str:
        if isinstance(self.name, str):
            return self.name
        else:
            return ".".join(self.name)

    @property
    def value(self) -> Any:
        """The object this name resolves to.

        For a dotted name this is the attribute, not the object holding it.
        Raises `ReferenceError` when the name is not in scope.
        """
        assert self.scope is not None

        try:
            if isinstance(self.name, str):
                return self.scope[self.name]
            else:
                name1, name2 = self.name
                obj = self.scope[name1]
                return obj.attrs[name2]
        except (KeyError, AttributeError):
            raise ReferenceError("Failed to resolve reference %s" % self, pos=self.pos)

    @property
    def values(self) -> list[Any]:
        """The chain this name resolves through, innermost last.

        One element for a plain name, and two for a dotted one,
        which lets a caller see the object as well as the attribute.
        Raises `ReferenceError` when the name is not in scope.
        """
        assert self.scope is not None

        try:
            if isinstance(self.name, str):
                return [self.scope[self.name]]
            else:
                name1, name2 = self.name
                obj = self.scope[name1]
                return [obj, obj.attrs[name2]]
        except (KeyError, AttributeError):
            raise ReferenceError("Failed to resolve reference %s" % self, pos=self.pos)


class UnaryExpr(AstNode):
    """A prefix operator applied to one operand.

    `type` is None until the type checker fills it in.
    """

    op: str
    arg: Expression
    type: str | None = None


class BinaryExpr(AstNode):
    """An infix operator applied to two operands.

    `type` is None until the type checker fills it in.
    """

    left: Expression
    op: str
    right: Expression
    type: str | None = None


class FuncCall(AstNode):
    """A call to a model function or to a builtin.

    `type` is None until the type checker fills it in.
    """

    func: Ref
    args: list[Expression]
    type: str | None = None


Expression = Literal | Ref | UnaryExpr | BinaryExpr | FuncCall


class LocalVariable(AstNode):
    """A variable introduced by an annotated assignment inside a function."""

    name: str
    type: TypeRef


class TypeRef(AstNode):
    """A type named in the source, which need not be a type that exists."""

    name: str


class PassStmt(AstNode):
    """A `pass` statement."""

    pass


class AssignmentStmt(AstNode):
    """An assignment, which carries the declared variable when it introduces one."""

    lvalue: Ref
    rvalue: Expression
    var: LocalVariable | None


class UpdateStmt(AstNode):
    """A compound assignment, such as `x += 1`."""

    lvalue: Ref
    op: str
    rvalue: Expression


class ReturnStmt(AstNode):
    """A `return` statement, linked to its enclosing function after parsing."""

    arg: Expression | None
    func: Func | None = Field(default=None, repr=False)


class IfStmt(AstNode):
    """An `if` statement with its `elif` sections and optional `else`."""

    condition: Expression
    block: Block
    elifs: list[ElifSection]
    else_: ElseSection | None


class ElifSection(AstNode):
    """One `elif` section of an `if` statement."""

    condition: Expression
    block: Block


class ElseSection(AstNode):
    """The `else` section of an `if` statement."""

    block: Block


Statement = PassStmt | AssignmentStmt | UpdateStmt | ReturnStmt | IfStmt


class Block(AstNode):
    """An indented run of statements."""

    stmts: list[Statement]


class Parameter(AstNode):
    """One declared parameter of a function."""

    name: str
    type: TypeRef


class Func(AstNode):
    """A `def` function.

    `lvars`, `return_stmts` and `scope` are empty
    until the parser's post-parse passes fill them in.
    """

    name: str
    params: list[Parameter]
    rtype: TypeRef | None
    block: Block
    lvars: list[LocalVariable] = Field(default_factory=list)
    return_stmts: list[ReturnStmt] = Field(default_factory=list)
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class Option(AstNode):
    """An `option` declaration, which is a compile-time constant."""

    name: str
    value: int


class Config(AstNode):
    """A `config` declaration, which becomes a simulator command line flag."""

    name: str
    type: TypeRef
    default: Expression


class TickVar(AstNode):
    """One variable of the `tick-data` block, and a column of the tick file."""

    name: str
    type: TypeRef
    annots: list[str]

    @property
    def is_real(self) -> bool:
        """Whether this carries data, rather than being the key column."""
        return "key" not in self.annots


class TickData(AstNode):
    """The `tick-data` block."""

    tick_vars: list[TickVar]

    @property
    def key_var(self) -> TickVar:
        """The variable annotated `key`, which indexes the tick file.

        Raises `CodeError` when the block declares none.
        """
        for var in self.tick_vars:
            if "key" in var.annots:
                return var

        raise CodeError("Key column not defined", pos=self.pos)


class TileVar(AstNode):
    """One variable of the `tile-data` block, and a grid over the tiles."""

    name: str
    type: TypeRef
    annots: list[str]

    @property
    def is_real(self) -> bool:
        """Whether this is stored per tile, rather than being the synthetic position."""
        return self.type.name != "position"

    @property
    def is_seeded(self) -> bool:
        """Whether this is read from the seed file rather than the tile file."""
        return "seed" in self.annots

    @property
    def is_saved(self) -> bool:
        """Whether this is written to the output file once per tick."""
        return "save" in self.annots


class TileData(AstNode):
    """The `tile-data` block."""

    tile_vars: list[TileVar]

    @property
    def state_var(self) -> TileVar:
        """The variable of type `fire_state`, which holds each tile's status.

        Raises `CodeError` when the block declares none.
        """
        for var in self.tile_vars:
            if var.type.name == "fire_state":
                return var

        raise CodeError("State column not defined", pos=self.pos)


class PoissonDist(AstNode):
    """A `poisson[mean]` distribution."""

    mean: Expression


class NormalDist(AstNode):
    """A `normal[mean, std]` distribution."""

    mean: Expression
    std: Expression


class DeterministicDist(AstNode):
    """A `deterministic[expr]` distribution, which samples to `expr` every time."""

    expr: Expression


class CreateEmbers(AstNode):
    """The `create-embers` clause, which binds one tile variable."""

    var_name: str
    dist: PoissonDist
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class EmberJumpLikelihood(AstNode):
    """The `ember-jump-likelihood` clause, binding a source and a destination tile."""

    svar_name: str
    dvar_name: str
    like: Expression
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class EmberDeathProb(AstNode):
    """The `ember-death-prob` clause, which binds a source and a destination tile."""

    svar_name: str
    dvar_name: str
    prob: Expression
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class EmberIgnitionProb(AstNode):
    """The `ember-ignition-prob` clause, which binds one tile variable."""

    var_name: str
    prob: Expression
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class CreateFlames(AstNode):
    """The `create-flames` clause, which binds one tile variable."""

    var_name: str
    dist: DeterministicDist
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class FlameSpreadWeight(AstNode):
    """The `flame-spread-weight` clause, which binds a source and a destination tile."""

    svar_name: str
    dvar_name: str
    weight: Expression
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class FlameIgnitionProb(AstNode):
    """The `flame-ignition-prob` clause, which binds one tile variable."""

    var_name: str
    prob: Expression
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class BurnTime(AstNode):
    """The `burn-time` clause, which binds one tile variable."""

    var_name: str
    dist: NormalDist
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class FireModel(AstNode):
    """The `fire-model` block, which holds all eight clauses."""

    create_embers: CreateEmbers
    ember_jump_likelihood: EmberJumpLikelihood
    ember_death_prob: EmberDeathProb
    ember_ignition_prob: EmberIgnitionProb
    create_flames: CreateFlames
    flame_spread_weight: FlameSpreadWeight
    flame_ignition_prob: FlameIgnitionProb
    burn_time: BurnTime


class Source(AstNode):
    """A whole FFSL source file, and the root of the tree.

    `opts` and `scope` are empty until the parser's post-parse passes fill them in.
    `opts` then holds every option with its default applied,
    so it is complete whether or not the source declared one.
    """

    options: list[Option]
    configs: list[Config]
    funcs: list[Func]
    tick_data: TickData
    tile_data: TileData
    fire_model: FireModel
    opts: dict[str, int] = Field(default_factory=dict, repr=False)
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)
