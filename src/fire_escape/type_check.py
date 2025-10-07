"""Type checking system."""

from __future__ import annotations

from typing import Self
from dataclasses import dataclass

import networkx as nx

from .ast_nodes import *
from .ast_nodes import Expression


@dataclass
class TypeEnv:
    graph: nx.DiGraph

    @classmethod
    def new(cls) -> Self:
        graph = nx.DiGraph()

        nx.add_path(graph, ("bool", "uint", "int", "float"))
        nx.add_path(graph, ("i8", "i16", "i32", "i64", "int"))
        nx.add_path(graph, ("u8", "u16", "u32", "u64", "uint"))
        nx.add_path(graph, ("f32", "f64", "float"))

        graph.add_node("str")

        env = cls(graph=graph)
        return env

    def is_convertable_to(self, child: str, ancestor: str) -> bool:
        if child == ancestor:
            return True

        if child in self.graph and ancestor in self.graph:
            return nx.has_path(self.graph, child, ancestor)
        else:
            return False

    def lub_type(self, type1: str, type2: str) -> str | None:
        if type1 in self.graph and type2 in self.graph:
            ltype = nx.lowest_common_ancestor(
                self.graph.reverse(copy=False), type1, type2
            )
            return ltype
        else:
            return None

    def check_unary(self, op: str, arg_type: str) -> str:
        match op:
            case "-":
                if not self.is_convertable_to(arg_type, "float"):
                    raise RuntimeError(f"Unary `-` not supported for type {arg_type}")
                return arg_type
            case "not":
                if not self.is_convertable_to(arg_type, "float"):
                    raise RuntimeError(f"Unary `not` not supported for type {arg_type}")
                return "bool"
            case _:
                raise RuntimeError(f"Unexpected unary operator: {op}")

    def check_binary(self, op: str, type1: str, type2: str) -> str:
        if op in ["or", "and"]:
            if not self.is_convertable_to(type1, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in ["==", "!="]:
            if type1 != type2:
                if not self.is_convertable_to(type1, "float"):
                    raise RuntimeError(f"Binary {op} not supported for type {type1}")
                if not self.is_convertable_to(type2, "float"):
                    raise RuntimeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in [">", ">=", "<", "<="]:
            if not self.is_convertable_to(type1, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in ["+", "-", "*", "/"]:
            if not self.is_convertable_to(type1, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {type2}")
            rtype = self.lub_type(type1, type2)
            assert rtype is not None
            return rtype
        elif op == "%":
            if not self.is_convertable_to(type1, "int"):
                raise RuntimeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "int"):
                raise RuntimeError(f"Binary {op} not supported for type {type2}")
            rtype = self.lub_type(type1, type2)
            assert rtype is not None
            return rtype
        elif op == "**":
            if not self.is_convertable_to(type1, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {type2}")
            return "float"
        else:
            raise RuntimeError(f"Unexpected binary operator: {op}")

    def check_assign(self, ltype: str, rtype: str):
        if not self.is_convertable_to(rtype, ltype):
            raise RuntimeError(
                f"Can't assign expression of type {ltype} to variable of type {rtype}"
            )

    def check_update(self, op: str, ltype: str, rtype: str):
        if op in ["+=", "-=", "*=", "/="]:
            if not self.is_convertable_to(ltype, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {ltype}")
            if not self.is_convertable_to(rtype, "float"):
                raise RuntimeError(f"Binary {op} not supported for type {rtype}")
        else:
            raise RuntimeError(f"Unexpected update operator: {op}")

        if not self.is_convertable_to(rtype, ltype):
            raise RuntimeError(
                f"Can't assign expression of type {ltype} to variable of type {rtype}"
            )


def get_type(node: Expression) -> str:
    match node:
        case Bool():
            return "bool"
        case Int():
            return "int"
        case Float():
            return "float"
        case Str():
            return "str"
        case Ref() as ref:
            obj = ref.value()
            match obj:
                case LocalVariable() as var:
                    return var.type.name
                case _:
                    raise RuntimeError(f"Unknown type {obj=}")
        case UnaryExpr() as expr:
            assert expr.type is not None
            return expr.type
        case BinaryExpr() as expr:
            assert expr.type is not None
            return expr.type
        case _ as unexpected:
            raise RuntimeError(f"{unexpected=}")


def check_type(node: AstNode, env: TypeEnv):
    for child in node.children:
        check_type(child, env)

    match node:
        case TypeRef() as tref:
            if tref.name not in env.graph:
                raise RuntimeError(f"Unknown type {tref.name}")
        case UnaryExpr() as expr:
            arg_type = get_type(expr.arg)
            expr.type = env.check_unary(expr.op, arg_type)
        case BinaryExpr() as expr:
            type1 = get_type(expr.left)
            type2 = get_type(expr.right)
            expr.type = env.check_binary(expr.op, type1, type2)
        case AssignmentStmt() as stmt:
            obj = stmt.lvalue.value()
            match obj:
                case LocalVariable() as var:
                    if var.type.is_const and stmt.var is None:
                        err = RuntimeError("Can't assign to constants")
                        err.add_note(str(stmt))
                        raise err
                    ltype = get_type(stmt.lvalue)
                    rtype = get_type(stmt.rvalue)
                    env.check_assign(ltype, rtype)
                case _:
                    raise RuntimeError(f"Unknown type {obj=}")
        case UpdateStmt() as stmt:
            obj = stmt.lvalue.value()
            match obj:
                case LocalVariable() as var:
                    if var.type.is_const:
                        err = RuntimeError("Can't assign to constants")
                        err.add_note(str(stmt))
                        raise err
                    ltype = get_type(stmt.lvalue)
                    rtype = get_type(stmt.rvalue)
                    env.check_update(stmt.op, ltype, rtype)
                case _:
                    raise RuntimeError(f"Unknown type {obj=}")
