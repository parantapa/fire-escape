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
]

from dataclasses import dataclass, field


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
class TypeRef(AstNode):
    is_const: bool
    name: str


@dataclass
class AssignmentStmt(AstNode):
    lvalue: Ref
    type: TypeRef | None
    rvalue: Expression


@dataclass
class UpdateStmt(AstNode):
    lvalue: Ref
    op: str
    rvalue: Expression


Statement = AssignmentStmt | UpdateStmt


@dataclass
class Source(AstNode):
    stmts: list[Statement]
