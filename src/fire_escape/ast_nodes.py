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
    "IfStmt",
    "ElifSection",
    "ElseSection",
    "Source",
    "LocalVariable",
    "Block"
]

from dataclasses import dataclass, field
from collections import ChainMap
from typing import Any

from .error import *


@dataclass
class AstNode:
    pos: Position = field(repr=False, compare=False)
    children: list[AstNode] = field(repr=False, compare=False)


@dataclass
class Bool(AstNode):
    value: bool


@dataclass
class Int(AstNode):
    value: int


@dataclass
class Float(AstNode):
    value: float


@dataclass
class Str(AstNode):
    value: str


Literal = Bool | Int | Float | Str


@dataclass
class Ref(AstNode):
    name: str
    scope: ChainMap[str, Any] | None = field(default=None, repr=False, compare=False)

    def value(self) -> Any:
        assert self.scope is not None

        try:
            return self.scope[self.name]
        except KeyError:
            raise ReferenceError(f"{self.name} not defined")


@dataclass
class UnaryExpr(AstNode):
    op: str
    arg: Expression
    type: str | None = field(default=None)


@dataclass
class BinaryExpr(AstNode):
    left: Expression
    op: str
    right: Expression
    type: str | None = field(default=None)


@dataclass
class FuncCall(AstNode):
    func: Ref
    args: list[Expression]
    type: str | None = field(default=None)


@dataclass
class JsonExpr(AstNode):
    jvar: Ref
    idxs: list[Expression]
    type: TypeRef


Expression = Literal | Ref | UnaryExpr | BinaryExpr | FuncCall | JsonExpr


@dataclass
class LocalVariable(AstNode):
    name: str
    type: TypeRef


@dataclass
class TypeRef(AstNode):
    is_const: bool
    name: str


@dataclass
class PassStmt(AstNode):
    pass


@dataclass
class AssignmentStmt(AstNode):
    lvalue: Ref
    rvalue: Expression
    var: LocalVariable | None


@dataclass
class UpdateStmt(AstNode):
    lvalue: Ref
    op: str
    rvalue: Expression


@dataclass
class PrintStmt(AstNode):
    args: list[Expression]


@dataclass
class IfStmt(AstNode):
    condition: Expression
    block: Block
    elifs: list[ElifSection]
    else_: ElseSection | None


@dataclass
class ElifSection(AstNode):
    condition: Expression
    block: Block


@dataclass
class ElseSection(AstNode):
    block: Block


Statement = PassStmt | AssignmentStmt | UpdateStmt | PrintStmt | IfStmt


@dataclass
class Block(AstNode):
    stmts: list[Statement]


@dataclass
class Source(AstNode):
    block: Block
    lvars: list[LocalVariable] = field(default_factory=list, compare=False)
    scope: ChainMap[str, Any] | None = field(default=None, repr=False, compare=False)
