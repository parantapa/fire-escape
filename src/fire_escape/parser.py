"""FFSL parser module."""

from __future__ import annotations

import importlib.resources
from functools import cache
from collections import ChainMap
from typing import Any, cast
from pathlib import Path

from lark import Lark, Tree, Token

from .ast_nodes import *
from .error import *
from .type_check import check_type, TypeEnv
from .builtins import add_builtins

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

    raise CompilerError("Unable to create parser.")


def _unary(children, pos):
    match children:
        case [op, arg]:
            return UnaryExpr(op=op.value, arg=arg, pos=pos, children=[arg])
        case [arg]:
            return arg
        case _ as unexpected:
            raise CompilerError(f"Unexpected unary expression: {unexpected=}")


def _binary_left_assoc(children, pos):
    if len(children) == 1:
        return children[0]
    else:
        *left, op, right = children
        left = _binary_left_assoc(left, pos)
        return BinaryExpr(
            left=left,
            op=op.value,
            right=right,
            pos=pos,
            children=[left, right],
        )


def build_ast(tree: Tree, file: str):
    children = [
        build_ast(child, file) if isinstance(child, Tree) else child
        for child in tree.children
    ]
    pos = Position(file=file, line=tree.meta.line, col=tree.meta.column)

    match tree.data:
        case "true":
            return Bool(value=True, pos=pos, children=[])

        case "false":
            return Bool(value=False, pos=pos, children=[])

        case "int":
            (child,) = children
            value = int(child.value)
            return Int(value=value, pos=pos, children=[])

        case "float":
            (child,) = children
            value = float(child.value)
            return Float(value=value, pos=pos, children=[])

        case "str":
            (child,) = children
            child = cast(Token, child)
            value = child.value.lstrip('"').rstrip('"')
            return Str(value=value, pos=pos, children=[])

        case "ref":
            (child,) = children
            name = child.value
            return Ref(name=name, pos=pos, children=[])

        case "unary_neg" | "unary_not":
            return _unary(children, pos)

        case "binary_exp" | "binary_mul" | "binary_add" | "binary_cmp" | "binary_and":
            return _binary_left_assoc(children, pos)

        case "func_call":
            func, *args = children
            return FuncCall(func=func, args=args, pos=pos, children=children)

        case "json_expr":
            jvar, *idxs, type = children
            return JsonExpr(
                jvar=jvar,
                idxs=idxs,
                type=type,
                pos=pos,
                children=children,
            )

        case "type":
            match children:
                case [op, name]:
                    return TypeRef(
                        is_const=True,
                        name=name.value,
                        pos=pos,
                        children=[],
                    )
                case [name]:
                    return TypeRef(
                        is_const=False,
                        name=name.value,
                        pos=pos,
                        children=[],
                    )
                case _ as unexpected:
                    raise CompilerError(f"Unexpected type: {unexpected=}")

        case "pass_stmt":
            return PassStmt(pos=pos, children=[])

        case "assignment_stmt":
            match children:
                case [lvalue, type, rvalue]:
                    var = LocalVariable(
                        name=lvalue.name,
                        type=type,
                        pos=pos,
                        children=[type],
                    )
                    return AssignmentStmt(
                        lvalue=lvalue,
                        rvalue=rvalue,
                        var=var,
                        pos=pos,
                        children=[lvalue, rvalue, var],
                    )
                case [lvalue, rvalue]:
                    return AssignmentStmt(
                        lvalue=lvalue,
                        rvalue=rvalue,
                        var=None,
                        pos=pos,
                        children=[lvalue, rvalue],
                    )
                case _ as unexpected:
                    raise CompilerError(f"Unexpected assignment_stmt: {unexpected=}")

        case "update_stmt":
            lvalue, op, rvalue = children
            return UpdateStmt(
                lvalue=lvalue,
                op=op.value,
                rvalue=rvalue,
                pos=pos,
                children=[lvalue, rvalue],
            )

        case "else_section":
            return ElseSection(stmts=children, pos=pos, children=children)

        case "elif_section":
            condition, *stmts = children
            return ElifSection(
                condition=condition,
                stmts=stmts,
                pos=pos,
                children=children,
            )

        case "if_stmt":
            condition, *rest = children
            stmts = []
            elifs = []
            else_ = None
            for obj in rest:
                match obj:
                    case ElseSection():
                        else_ = obj
                    case ElifSection():
                        elifs.append(obj)
                    case _:
                        stmts.append(obj)
            return IfStmt(
                condition=condition,
                stmts=stmts,
                elifs=elifs,
                else_=else_,
                pos=pos,
                children=children,
            )

        case "print_stmt":
            return PrintStmt(args=children, pos=pos, children=children)

        case "source":
            return Source(stmts=children, pos=pos, children=children)

        case _ as unexpected:
            raise CompilerError(f"unexpected tree.data={unexpected}; {children=}")


def build_scope(node: AstNode, scope: ChainMap[str, Any] | None):
    try:
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
                    raise ReferenceError(
                        f"{var.name} has already been defined.",
                        pos=var.pos,
                    )
                scope[var.name] = var

        for child in node.children:
            build_scope(child, scope)
    except CodeError as e:
        e.pos = node.pos if e.pos is None else e.pos
        raise e


def collect_local_varaibles(node: AstNode):
    try:
        match node:
            case Source() as source:
                assert source.scope is not None
                for var in source.scope.maps[0].values():
                    if isinstance(var, LocalVariable):
                        source.lvars.append(var)

        for child in node.children:
            collect_local_varaibles(child)
    except CodeError as e:
        e.pos = node.pos if e.pos is None else e.pos
        raise e


def parse(file: str, text: str):
    parser = get_parser()
    tree = parser.parse(text)
    source: Source = cast(Source, build_ast(tree, file))
    build_scope(source, None)
    collect_local_varaibles(source)

    assert source.scope is not None
    add_builtins(source.scope)

    env = TypeEnv.new()
    check_type(source, env)
    return source
