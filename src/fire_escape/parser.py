"""FFSL parser module."""

from __future__ import annotations

import importlib.resources
from functools import cache
from collections import ChainMap
from typing import Any

from lark import Lark, Transformer

from .ast_nodes import *
from .type_check import check_type, TypeEnv

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


class AstTransformer(Transformer):
    def true(self, children):
        child = children[0]
        return Bool(value=True, line=child.line, col=child.column, children=[])

    def false(self, children):
        child = children[0]
        return Bool(value=False, line=child.line, col=child.column, children=[])

    def int(self, children):
        child = children[0]
        value = int(child.value)
        return Int(value=value, line=child.line, col=child.column, children=[])

    def float(self, children):
        child = children[0]
        value = float(child.value)
        return Float(value=value, line=child.line, col=child.column, children=[])

    def str(self, children):
        child = children[0]
        value = child.value.lstrip('"').rstrip('"')
        return Str(value=value, line=child.line, col=child.column, children=[])

    def ref(self, children):
        child = children[0]
        name = child.value
        return Ref(name=name, line=child.line, col=child.column, children=[])

    # def primary(self, children):
    #     return children[0]

    def _unary(self, children):
        match children:
            case op, arg:
                return UnaryExpr(
                    op=op.value, arg=arg, line=op.line, col=op.column, children=[arg]
                )
            case arg:
                return arg

    def _binary_left_assoc(self, children):
        if len(children) == 1:
            return children[0]
        else:
            *left, op, right = children
            left = self._binary_left_assoc(left)
            return BinaryExpr(
                left=left,
                op=op.value,
                right=right,
                line=left.line,
                col=left.col,
                children=[left, right],
            )

    unary_neg = _unary
    unary_not = _unary

    binary_exp = _binary_left_assoc
    binary_mul = _binary_left_assoc
    binary_add = _binary_left_assoc
    binary_cmp = _binary_left_assoc
    binary_and = _binary_left_assoc

    def type(self, children):
        match children:
            case [op, name]:
                return TypeRef(
                    is_const=True,
                    name=name.value,
                    line=op.line,
                    col=op.column,
                    children=[],
                )
            case [name]:
                return TypeRef(
                    is_const=False,
                    name=name.value,
                    line=name.line,
                    col=name.column,
                    children=[],
                )
            case _ as unexpected:
                raise RuntimeError(f"{unexpected=}")

    def assignment_stmt(self, children):
        match children:
            case [lvalue, type, rvalue]:
                var = LocalVariable(
                    name=lvalue.name,
                    type=type,
                    line=lvalue.line,
                    col=lvalue.col,
                    children=[type],
                )
                return AssignmentStmt(
                    lvalue=lvalue,
                    rvalue=rvalue,
                    var=var,
                    line=lvalue.line,
                    col=lvalue.col,
                    children=[lvalue, rvalue, var],
                )
            case [lvalue, rvalue]:
                return AssignmentStmt(
                    lvalue=lvalue,
                    rvalue=rvalue,
                    var=None,
                    line=lvalue.line,
                    col=lvalue.col,
                    children=[lvalue, rvalue],
                )
            case _ as unexpected:
                raise RuntimeError(f"{unexpected=}")

    def update_stmt(self, children):
        match children:
            case [lvalue, op, rvalue]:
                return UpdateStmt(
                    lvalue=lvalue,
                    op=op.value,
                    rvalue=rvalue,
                    line=lvalue.line,
                    col=lvalue.col,
                    children=[lvalue, rvalue],
                )
            case _ as unexpected:
                raise RuntimeError(f"{unexpected=}")

    def source(self, children):
        child = children[0]
        return Source(stmts=children, line=child.line, col=child.col, children=children)


def build_scope(node: AstNode, scope: ChainMap[str, Any] | None):
    match node:
        case Source() as source:
            assert scope is None
            scope = ChainMap()
            source.scope = scope
        case Ref() as ref:
            assert scope is not None
            ref.scope = scope
        case LocalVariable() as var:
            assert scope is not None
            if var.name in scope.maps[0]:
                raise RuntimeError(f"{var.name} has already been defined.")
            scope[var.name] = var

    for child in node.children:
        build_scope(child, scope)


def parse(text: str):
    parser = get_parser()
    tree = parser.parse(text)
    source: Source = AstTransformer().transform(tree)
    build_scope(source, None)

    env = TypeEnv.new()
    check_type(source, env)
    return source
