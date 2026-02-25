"""AST node definitions."""

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
    "CreateEmbers",
    "EmberJumpLikelihood",
    "EmberDeathProb",
    "IgnitionProb",
    "BurnTime",
    "FireModel",
]

from collections import ChainMap
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .error import *


class AstNode(BaseModel):
    pos: Position = Field(repr=False)
    children: list[AstNode] = Field(repr=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Bool(AstNode):
    value: bool


class Int(AstNode):
    value: int


class Float(AstNode):
    value: float


class Str(AstNode):
    value: str


Literal = Bool | Int | Float | Str


class Ref(AstNode):
    name: str | tuple[str, str]
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)

    def __str__(self) -> str:
        if isinstance(self.name, str):
            return self.name
        else:
            return ".".join(self.name)

    @property
    def value(self) -> Any:
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
    op: str
    arg: Expression
    type: str | None = None


class BinaryExpr(AstNode):
    left: Expression
    op: str
    right: Expression
    type: str | None = None


class FuncCall(AstNode):
    func: Ref
    args: list[Expression]
    type: str | None = None


Expression = Literal | Ref | UnaryExpr | BinaryExpr | FuncCall


class LocalVariable(AstNode):
    name: str
    type: TypeRef


class TypeRef(AstNode):
    name: str


class PassStmt(AstNode):
    pass


class AssignmentStmt(AstNode):
    lvalue: Ref
    rvalue: Expression
    var: LocalVariable | None


class UpdateStmt(AstNode):
    lvalue: Ref
    op: str
    rvalue: Expression


class ReturnStmt(AstNode):
    arg: Expression | None
    func: Func | None = Field(default=None, repr=False)


class IfStmt(AstNode):
    condition: Expression
    block: Block
    elifs: list[ElifSection]
    else_: ElseSection | None


class ElifSection(AstNode):
    condition: Expression
    block: Block


class ElseSection(AstNode):
    block: Block


Statement = PassStmt | AssignmentStmt | UpdateStmt | ReturnStmt | IfStmt


class Block(AstNode):
    stmts: list[Statement]


class Parameter(AstNode):
    name: str
    type: TypeRef


class Func(AstNode):
    name: str
    params: list[Parameter]
    rtype: TypeRef | None
    block: Block
    lvars: list[LocalVariable] = Field(default_factory=list)
    return_stmts: list[ReturnStmt] = Field(default_factory=list)
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class Option(AstNode):
    name: str
    value: int


class Config(AstNode):
    name: str
    type: TypeRef
    default: Expression


class TickVar(AstNode):
    name: str
    type: TypeRef
    annots: list[str]

    @property
    def is_real(self) -> bool:
        return "key" not in self.annots


class TickData(AstNode):
    tick_vars: list[TickVar]

    @property
    def key_var(self) -> TickVar:
        for var in self.tick_vars:
            if "key" in var.annots:
                return var

        raise CodeError("Key column not defined", pos=self.pos)


class TileVar(AstNode):
    name: str
    type: TypeRef
    annots: list[str]

    @property
    def is_real(self) -> bool:
        return self.type.name != "position"


class TileData(AstNode):
    tile_vars: list[TileVar]

    @property
    def state_var(self) -> TileVar:
        for var in self.tile_vars:
            if var.type.name == "fire_state":
                return var

        raise CodeError("State column not defined", pos=self.pos)


class PoissonDist(AstNode):
    mean: Expression


class NormalDist(AstNode):
    mean: Expression
    std: Expression


class CreateEmbers(AstNode):
    var_name: str
    dist: PoissonDist
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class EmberJumpLikelihood(AstNode):
    svar_name: str
    dvar_name: str
    like: Expression
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class EmberDeathProb(AstNode):
    svar_name: str
    dvar_name: str
    prob: Expression
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class IgnitionProb(AstNode):
    var_name: str
    prob: Expression
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class BurnTime(AstNode):
    var_name: str
    dist: NormalDist
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class FireModel(AstNode):
    create_embers: CreateEmbers
    ember_jump_likelihood: EmberJumpLikelihood
    ember_death_prob: EmberDeathProb
    ignition_prob: IgnitionProb
    burn_time: BurnTime


class Source(AstNode):
    options: list[Option]
    configs: list[Config]
    funcs: list[Func]
    tick_data: TickData
    tile_data: TileData
    fire_model: FireModel
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)
