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
    "JsonExpr",
    "TypeRef",
    "PassStmt",
    "AssignmentStmt",
    "UpdateStmt",
    "PrintStmt",
    "ReturnStmt",
    "IfStmt",
    "ElifSection",
    "ElseSection",
    "Parameter",
    "Func",
    "Source",
    "LocalVariable",
    "Block",
]

from pydantic import BaseModel, ConfigDict, Field
from collections import ChainMap
from typing import Any

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
    name: str
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)

    def value(self) -> Any:
        assert self.scope is not None

        try:
            return self.scope[self.name]
        except KeyError:
            raise ReferenceError(f"{self.name} not defined")


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


class JsonExpr(AstNode):
    jvar: Ref
    idxs: list[Expression]
    type: TypeRef


Expression = Literal | Ref | UnaryExpr | BinaryExpr | FuncCall | JsonExpr


class LocalVariable(AstNode):
    name: str
    type: TypeRef


class TypeRef(AstNode):
    is_const: bool
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


class PrintStmt(AstNode):
    args: list[Expression]


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


Statement = PassStmt | AssignmentStmt | UpdateStmt | PrintStmt | ReturnStmt | IfStmt


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
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)


class Source(AstNode):
    funcs: list[Func]
    scope: ChainMap[str, Any] | None = Field(default=None, repr=False)
