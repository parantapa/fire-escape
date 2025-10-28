"""FFSL parser module."""

from __future__ import annotations

import importlib.resources
from functools import cache
from collections import ChainMap
from typing import Any, cast

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
            raise CompilerError(f"{unexpected=}")


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
        if child is not None
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
                    raise CompilerError(f"{unexpected=}")

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
                    raise CompilerError(f"{unexpected=}")

        case "update_stmt":
            lvalue, op, rvalue = children
            return UpdateStmt(
                lvalue=lvalue,
                op=op.value,
                rvalue=rvalue,
                pos=pos,
                children=[lvalue, rvalue],
            )

        case "block":
            return Block(stmts=children, pos=pos, children=children)

        case "else_section":
            return ElseSection(block=children[0], pos=pos, children=children)

        case "elif_section":
            condition, block = children
            return ElifSection(
                condition=condition,
                block=block,
                pos=pos,
                children=children,
            )

        case "if_stmt":
            condition, block, *rest = children
            elifs = []
            else_ = None
            for obj in rest:
                match obj:
                    case ElseSection():
                        else_ = obj
                    case ElifSection():
                        elifs.append(obj)
                    case _ as unexpected:
                        raise CompilerError(f"{unexpected=}")
            return IfStmt(
                condition=condition,
                block=block,
                elifs=elifs,
                else_=else_,
                pos=pos,
                children=children,
            )

        case "print_stmt":
            return PrintStmt(args=children, pos=pos, children=children)

        case "return_stmt":
            if children:
                return ReturnStmt(arg=children[0], pos=pos, children=children)
            else:
                return ReturnStmt(arg=None, pos=pos, children=[])

        case "param":
            name, type = children
            return Parameter(name=name.value, type=type, pos=pos, children=children)

        case "func":
            name, *rest = children
            name = name.value

            params = []
            rtype = None
            block = None
            for obj in rest:
                match obj:
                    case Parameter():
                        params.append(obj)
                    case TypeRef():
                        assert rtype is None
                        rtype = obj
                    case Block():
                        assert block is None
                        block = obj
                    case _ as unexpected:
                        raise CompilerError(f"{unexpected=}")

            assert block is not None

            children = list(params)
            if rtype is not None:
                children.append(rtype)
            children.append(block)

            return Func(
                name=name,
                params=params,
                rtype=rtype,
                block=block,
                pos=pos,
                children=children,
            )

        case "source":
            return Source(funcs=children, pos=pos, children=children)

        case _ as unexpected:
            raise CompilerError(f"unexpected tree.data={unexpected}; {children=}")


def build_scope(node: AstNode, scope: ChainMap[str, Any] | None):
    try:
        match node:
            case Source() as source:
                assert scope is None
                scope = ChainMap()
                source.scope = scope
            case Func() as func:
                assert scope is not None

                if func.name in scope.maps[0]:
                    raise ReferenceError(
                        f"{func.name} has already been defined.",
                        pos=func.pos,
                    )
                scope[func.name] = func

                scope = scope.new_child()
                func.scope = scope
            case Ref() as ref:
                assert scope is not None
                ref.scope = scope
            case LocalVariable() | Parameter() as var:
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
    except CompilerError:
        raise
    except Exception as e:
        raise CompilerError(f"{node=} {scope=}") from e


def collect_local_varaibles(node: AstNode):
    try:
        match node:
            case Func() as func:
                assert func.scope is not None
                for var in func.scope.maps[0].values():
                    if isinstance(var, LocalVariable):
                        func.lvars.append(var)

        for child in node.children:
            collect_local_varaibles(child)
    except CodeError as e:
        e.pos = node.pos if e.pos is None else e.pos
        raise e
    except CompilerError:
        raise
    except Exception as e:
        raise CompilerError(f"{node=}") from e


def link_return_statements(node: AstNode, func: Func | None):
    try:
        match node:
            case Func() as func:
                func = func
            case ReturnStmt() as stmt:
                assert func is not None
                assert stmt.func is None
                stmt.func = func
                func.return_stmts.append(stmt)

        for child in node.children:
            link_return_statements(child, func)
    except CodeError as e:
        e.pos = node.pos if e.pos is None else e.pos
        raise e
    except CompilerError:
        raise
    except Exception as e:
        raise CompilerError(f"{node=}") from e


def parse(file: str, text: str):
    parser = get_parser()
    tree = parser.parse(text)
    source: Source = cast(Source, build_ast(tree, file))

    build_scope(source, None)
    assert source.scope is not None

    collect_local_varaibles(source)
    link_return_statements(source, None)
    add_builtins(source.scope)

    env = TypeEnv.new()
    check_type(source, env)
    return source
