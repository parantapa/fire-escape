"""Type checking system."""

from __future__ import annotations

from typing import Self
from dataclasses import dataclass

import networkx as nx

from .ast_nodes import *
from .builtins import *
from .error import *


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
                    raise TypeError(f"Unary `-` not supported for type {arg_type}")
                if self.is_convertable_to(arg_type, "int"):
                    return "int"
                else:
                    return "float"
            case "not":
                if not self.is_convertable_to(arg_type, "float"):
                    raise TypeError(f"Unary `not` not supported for type {arg_type}")
                return "bool"
            case _:
                raise CompilerError(f"Unexpected unary operator: {op}")

    def check_binary(self, op: str, type1: str, type2: str) -> str:
        if op in ["or", "and"]:
            if not self.is_convertable_to(type1, "float"):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "float"):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in ["==", "!="]:
            if type1 != type2:
                if not self.is_convertable_to(type1, "float"):
                    raise TypeError(f"Binary {op} not supported for type {type1}")
                if not self.is_convertable_to(type2, "float"):
                    raise TypeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in [">", ">=", "<", "<="]:
            if not self.is_convertable_to(type1, "float"):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "float"):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in ["+", "-", "*", "/"]:
            if not self.is_convertable_to(type1, "float"):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "float"):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            rtype = self.lub_type(type1, type2)
            assert rtype is not None
            return rtype
        elif op == "%":
            if not self.is_convertable_to(type1, "int"):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "int"):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            rtype = self.lub_type(type1, type2)
            assert rtype is not None
            return rtype
        elif op == "**":
            if not self.is_convertable_to(type1, "float"):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_convertable_to(type2, "float"):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            return "float"
        else:
            raise CompilerError(f"Unexpected binary operator: {op}")

    def check_call(self, ptypes: list[str], rtype: str, atypes: list[str]) -> str:
        if not len(ptypes) == len(atypes):
            raise TypeError(
                f"Parameter count mismatch: expected {len(ptypes)}, got {len(atypes)}"
            )
        for i, (ptype, atype) in enumerate(zip(ptypes, atypes), 1):
            if not self.is_convertable_to(atype, ptype):
                raise TypeError(
                    f"Parameter count mismatch: argument {i}: Can't convert argument of type {atype} to {ptype}"
                )
        return rtype

    def check_assign(self, ltype: str, rtype: str):
        if not self.is_convertable_to(rtype, ltype):
            raise TypeError(
                f"Can't assign expression of type {ltype} to variable of type {rtype}"
            )

    def check_update(self, op: str, ltype: str, rtype: str):
        if op in ["+=", "-=", "*=", "/="]:
            if not self.is_convertable_to(ltype, "float"):
                raise TypeError(f"Binary {op} not supported for type {ltype}")
            if not self.is_convertable_to(rtype, "float"):
                raise TypeError(f"Binary {op} not supported for type {rtype}")
        else:
            raise CompilerError(f"Unexpected update operator: {op}")

        if not self.is_convertable_to(rtype, ltype):
            raise TypeError(
                f"Can't assign expression of type {ltype} to variable of type {rtype}"
            )


def get_type(node: AstNode) -> str:
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
                case BuiltinFunc() as fn:
                    return fn.type
                case _:
                    raise RuntimeError(f"Unknown type {obj=}")
        case UnaryExpr() as expr:
            assert expr.type is not None
            return expr.type
        case BinaryExpr() as expr:
            assert expr.type is not None
            return expr.type
        case FuncCall() as call:
            assert call.type is not None
            return call.type
        case _ as unexpected:
            raise CompilerError(f"Unexpected expression type: {unexpected=}")


def check_type(node: AstNode, env: TypeEnv):
    try:
        for child in node.children:
            check_type(child, env)

        match node:
            case TypeRef() as tref:
                if tref.name not in env.graph:
                    raise TypeError(f"Unknown type: {tref.name}")
            case UnaryExpr() as expr:
                arg_type = get_type(expr.arg)
                expr.type = env.check_unary(expr.op, arg_type)
            case BinaryExpr() as expr:
                type1 = get_type(expr.left)
                type2 = get_type(expr.right)
                expr.type = env.check_binary(expr.op, type1, type2)
            case FuncCall() as call:
                atypes = [get_type(arg) for arg in call.args]
                match call.func.value():
                    case BuiltinFunc() as fn:
                        rtype = env.check_call(fn.ptypes, fn.rtype, atypes)
                        call.type = rtype
                    case _ as unexpected:
                        raise TypeError(f"{unexpected} is not callable")
            case AssignmentStmt() as stmt:
                obj = stmt.lvalue.value()
                match obj:
                    case LocalVariable() as var:
                        if var.type.is_const and stmt.var is None:
                            raise TypeError("Can't assign to constants")
                        ltype = get_type(stmt.lvalue)
                        rtype = get_type(stmt.rvalue)
                        env.check_assign(ltype, rtype)
                    case _:
                        raise CompilerError(f"Unknown lvalue type: {obj=}")
            case UpdateStmt() as stmt:
                obj = stmt.lvalue.value()
                match obj:
                    case LocalVariable() as var:
                        if var.type.is_const:
                            raise TypeError("Can't assign to constants")
                        ltype = get_type(stmt.lvalue)
                        rtype = get_type(stmt.rvalue)
                        env.check_update(stmt.op, ltype, rtype)
                    case _:
                        raise CompilerError(f"Unknown lvalue type: {obj=}")
            case IfStmt() as stmt:
                cond_type = get_type(stmt.condition)
                if not env.is_convertable_to(cond_type, "float"):
                    raise TypeError("Testexpression type not boolean or numeric")
            case ElifSection() as stmt:
                cond_type = get_type(stmt.condition)
                if not env.is_convertable_to(cond_type, "float"):
                    raise TypeError("Condition expression type not boolean or numeric")

    except CodeError as e:
        e.line = node.line if e.line is None else e.line
        e.col = node.col if e.col is None else e.col
        raise e
