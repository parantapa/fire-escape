"""FFSL parser module."""

from __future__ import annotations

import importlib.resources
from functools import cache
from dataclasses import dataclass, field

from lark import Lark, Transformer

GRAMMAR_ANCHOR = __name__
GRAMMAR_FILE = "ffsl.lark"


@cache
def get_parser() -> Lark:
    with importlib.resources.path(GRAMMAR_ANCHOR, GRAMMAR_FILE) as path:
        if path.exists():
            return Lark(
                path.read_text(),
                parser="lalr",
                start="source",
                strict=True,
                propagate_positions=True,
            )

    raise RuntimeError("Unable to create parser.")


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
class UnaryExpression(AstNode):
    op: str
    arg: Expression


@dataclass
class BinaryExpression(AstNode):
    left: Expression
    op: str
    right: Expression


Expression = Literal | UnaryExpression | BinaryExpression


class AstTransformer(Transformer):
    def true(self, children):
        child = children[0]
        return Bool(value=True, line=child.line, col=child.column)

    def false(self, children):
        child = children[0]
        return Bool(value=False, line=child.line, col=child.column)

    def int(self, children):
        child = children[0]
        value = int(child.value)
        return Int(value=value, line=child.line, col=child.column)

    def float(self, children):
        child = children[0]
        value = float(child.value)
        return Float(value=value, line=child.line, col=child.column)

    def str(self, children):
        child = children[0]
        value = child.value.lstrip('"').rstrip('"')
        return Str(value=value, line=child.line, col=child.column)

    def ref(self, children):
        child = children[0]
        name = child.value
        return Ref(name=name, line=child.line, col=child.column)

    # def primary(self, children):
    #     return children[0]

    def _unary(self, children):
        match children:
            case op, arg:
                return UnaryExpression(
                    op=op.value, arg=arg, line=op.line, col=op.column
                )
            case arg:
                return arg

    def _binary(self, children):
        if len(children) == 1:
            return children[0]
        else:
            left, op, *right = children
            right = self._binary(right)
            return BinaryExpression(
                left=left, op=op.value, right=right, line=left.line, col=left.col
            )

    unary_exp = _unary
    unary_neg = _unary
    unary_not = _unary

    binary_mul = _binary
    binary_add = _binary
    binary_cmp = _binary
    binary_and = _binary


def parse(text: str):
    parser = get_parser()
    tree = parser.parse(text)
    tree = AstTransformer().transform(tree)
    return tree
