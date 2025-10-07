"""AST node definitions."""

from __future__ import annotations

__all__ = ["AstNode", "Bool", "Int", "Float", "Str", "Ref", "UnaryExpr", "BinaryExpr"]

from dataclasses import dataclass, field


@dataclass
class AstNode:
    line: int = field(repr=False, compare=False)
    col: int = field(repr=False, compare=False)


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


@dataclass
class Ref(AstNode):
    name: str


Literal = Int | Float | Str | Ref


@dataclass
class UnaryExpr(AstNode):
    op: str
    arg: Expression


@dataclass
class BinaryExpr(AstNode):
    left: Expression
    op: str
    right: Expression


Expression = Literal | UnaryExpr | BinaryExpr
