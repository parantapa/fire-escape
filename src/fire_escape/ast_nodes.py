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
    "Source",
    "LocalVariable",
]

from dataclasses import dataclass, field

from .scope import Scope


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
    scope: Scope | None = field(default=None)


@dataclass
class UnaryExpr(AstNode):
    op: str
    arg: Expression


@dataclass
class BinaryExpr(AstNode):
    left: Expression
    op: str
    right: Expression


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
class AssignmentStmt(AstNode):
    lvalue: Ref
    rvalue: Expression
    var: LocalVariable | None


@dataclass
class UpdateStmt(AstNode):
    lvalue: Ref
    op: str
    rvalue: Expression


Statement = AssignmentStmt | UpdateStmt


@dataclass
class Source(AstNode):
    stmts: list[Statement]
    scope: Scope | None = field(default=None)
