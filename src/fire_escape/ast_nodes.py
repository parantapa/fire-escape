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
    "TypeRef",
    "AssignmentStmt",
    "UpdateStmt",
    "PrintStmt",
    "Source",
    "LocalVariable",
]

from dataclasses import dataclass, field
from collections import ChainMap
from typing import Any


@dataclass
class AstNode:
    line: int = field(repr=False, compare=False)
    col: int = field(repr=False, compare=False)
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
        return self.scope[self.name]


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


Expression = Literal | Ref | UnaryExpr | BinaryExpr


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


Statement = PassStmt | AssignmentStmt | UpdateStmt | PrintStmt


@dataclass
class Source(AstNode):
    stmts: list[Statement]
    lvars: list[LocalVariable] = field(default_factory=list, compare=False)
    scope: ChainMap[str, Any] | None = field(default=None, repr=False, compare=False)
