"""The FFSL type lattice, and the pass that annotates the AST with types."""

from __future__ import annotations

from typing import Self
from dataclasses import dataclass

import networkx as nx

from .ast_nodes import *
from .builtins import *
from .error import *


@dataclass
class TypeEnv:
    """The conversion lattice over FFSL type names.

    An edge runs from a type to one it converts to implicitly,
    which can lose range or precision, as `i64` to `int` does.
    Reachability is therefore convertibility.
    """

    graph: nx.DiGraph

    @classmethod
    def new(cls) -> Self:
        """Build the environment holding the language's own types."""
        graph = nx.DiGraph()

        nx.add_path(graph, ("bool", "uint", "int", "float"))
        nx.add_path(graph, ("i8", "i16", "i32", "i64", "int"))
        nx.add_path(graph, ("u8", "u16", "u32", "u64", "uint"))
        nx.add_path(graph, ("f32", "f64", "float"))

        graph.add_node("position")
        graph.add_node("fire_state")

        env = cls(graph=graph)
        return env

    def is_convertable_to(self, child: str, ancestor: str) -> bool:
        """Whether `child` converts to `ancestor`, which every type does to itself.

        An unknown type name converts only to itself, rather than raising.
        """
        if child == ancestor:
            return True

        if child in self.graph and ancestor in self.graph:
            return nx.has_path(self.graph, child, ancestor)
        else:
            return False

    def is_numeric(self, child: str) -> bool:
        """Whether `child` converts to `float`, which `bool` does."""
        return self.is_convertable_to(child, "float")

    def is_integral(self, child: str) -> bool:
        """Whether `child` converts to `int`, which `bool` does."""
        return self.is_convertable_to(child, "int")

    def lub_type(self, type1: str, type2: str) -> str | None:
        """Return the narrowest type both arguments convert to.

        `None` when either type is unknown,
        or when the two have no common type, as `position` and `int` do.
        """
        if type1 in self.graph and type2 in self.graph:
            # A lowest common ancestor in the reversed graph is a join.
            ltype = nx.lowest_common_ancestor(
                self.graph.reverse(copy=False), type1, type2
            )
            return ltype
        else:
            return None

    def check_unary(self, op: str, arg_type: str) -> str:
        """Return the result type of `op arg_type`.

        Raises `TypeError` when the operand type does not support `op`,
        and `CompilerError` for an operator the grammar should not produce.
        """
        match op:
            case "-":
                if not self.is_numeric(arg_type):
                    raise TypeError(f"Unary `-` not supported for type {arg_type}")
                if self.is_integral(arg_type):
                    return "int"
                else:
                    return "float"
            case "not":
                if not self.is_numeric(arg_type):
                    raise TypeError(f"Unary `not` not supported for type {arg_type}")
                return "bool"
            case _:
                raise CompilerError(f"Unexpected unary operator: {op}")

    def check_binary(self, op: str, type1: str, type2: str) -> str:
        """Return the result type of `type1 op type2`.

        Raises `TypeError` when either operand type does not support `op`,
        and `CompilerError` for an operator the grammar should not produce.
        """
        if op in ["or", "and"]:
            if not self.is_numeric(type1):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_numeric(type2):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in ["==", "!="]:
            if type1 != type2:
                if not self.is_numeric(type1):
                    raise TypeError(f"Binary {op} not supported for type {type1}")
                if not self.is_numeric(type2):
                    raise TypeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in [">", ">=", "<", "<="]:
            if not self.is_numeric(type1):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_numeric(type2):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            return "bool"
        elif op in ["+", "-", "*"]:
            if not self.is_numeric(type1):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_numeric(type2):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            # Both operands reach `float`, so a join always exists.
            rtype = self.lub_type(type1, type2)
            assert rtype is not None
            return rtype
        elif op == "/":
            # `/` is always true division.
            # See "Division is always true division" in docs/developer-notes.md.
            if not self.is_numeric(type1):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_numeric(type2):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            return "float"
        elif op == "%":
            if not self.is_integral(type1):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_integral(type2):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            # Both operands reach `int`, so a join always exists.
            rtype = self.lub_type(type1, type2)
            assert rtype is not None
            return rtype
        elif op == "**":
            if not self.is_numeric(type1):
                raise TypeError(f"Binary {op} not supported for type {type1}")
            if not self.is_numeric(type2):
                raise TypeError(f"Binary {op} not supported for type {type2}")
            return "float"
        else:
            raise CompilerError(f"Unexpected binary operator: {op}")

    def check_func_call(self, ptypes: list[str], rtype: str, atypes: list[str]) -> str:
        """Check the arguments against the parameters, then return `rtype`.

        Raises `TypeError` on an arity mismatch,
        or when an argument type does not convert to its parameter type.
        """
        if not len(ptypes) == len(atypes):
            raise TypeError(
                f"Parameter count mismatch: expected {len(ptypes)}, got {len(atypes)}"
            )
        for i, (ptype, atype) in enumerate(zip(ptypes, atypes), 1):
            if not self.is_convertable_to(atype, ptype):
                raise TypeError(
                    f"Parameter mismatch: argument {i}: Can't convert argument of type {atype} to {ptype}"
                )
        return rtype

    def check_assign(self, ltype: str, rtype: str) -> None:
        """Raise `TypeError` unless a value of `rtype` can be stored in `ltype`."""
        if not self.is_convertable_to(rtype, ltype):
            raise TypeError(
                f"Can't assign expression of type {rtype} to variable of type {ltype}"
            )

    def check_update(self, op: str, ltype: str, rtype: str) -> None:
        """Raise `TypeError` unless `ltype op= rtype` is well typed.

        `/=` on an integral left operand is rejected,
        because true division yields a float.

        Raises `CompilerError` for an operator the grammar should not produce.
        """
        if op not in ["+=", "-=", "*=", "/="]:
            raise CompilerError(f"Unexpected update operator: {op}")

        if not self.is_numeric(ltype):
            raise TypeError(f"Binary {op} not supported for type {ltype}")
        if not self.is_numeric(rtype):
            raise TypeError(f"Binary {op} not supported for type {rtype}")

        # An update statement is the matching binary operation,
        # followed by an assignment back to the left operand.
        # The result of that operation therefore drives the check,
        # which is what keeps `/=` in step with true division.
        result = self.check_binary(op[0], ltype, rtype)
        if not self.is_convertable_to(result, ltype):
            raise TypeError(
                f"Can't assign expression of type {result} to variable of type {ltype}"
            )


def get_type(node: AstNode | BuiltinFunc | BuiltinObject) -> str:
    """Return the FFSL type name of an already checked node.

    `check_type` must run over an expression node first,
    since its type is read off the annotation that pass writes.
    A callable yields its signature rendered as a string.

    Raises `CompilerError` for a node kind that has no type.
    An expression node that `check_type` has not annotated
    fails an assertion instead.
    """
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
            return get_type(ref.value)

        case UnaryExpr() | BinaryExpr() | FuncCall() as expr:
            assert expr.type is not None
            return expr.type

        case TypeRef() as tref:
            return tref.name

        case LocalVariable() | Parameter() as var:
            return get_type(var.type)

        case BuiltinFunc() as func:
            return func.type

        case BuiltinObject() as obj:
            return obj.type

        case Func() as func:
            ptypes = [get_type(param) for param in func.params]
            ptypes = ", ".join(ptypes)
            rtype = "void" if func.rtype is None else get_type(func.rtype)
            return f"({ptypes}) -> {rtype}"

        case Config() | TileVar() | TickVar() as var:
            return get_type(var.type)

        case _ as unexpected:
            raise CompilerError(f"Unexpected expression type: {unexpected=}")


def block_terminates(block: Block) -> bool:
    """Return True when control cannot fall off the end of the block.

    A block terminates as soon as one of its statements terminates.
    Statements after that one are unreachable.
    """
    return any(stmt_terminates(stmt) for stmt in block.stmts)


def stmt_terminates(stmt: AstNode) -> bool:
    """Return True when control cannot flow past the statement.

    A return statement always terminates.
    An if statement terminates when it has an else section,
    and every one of its branches terminates.
    An if statement without an else section never terminates.
    """
    match stmt:
        case ReturnStmt():
            return True

        case IfStmt() as stmt:
            # The condition can be false, so control can skip every branch.
            if stmt.else_ is None:
                return False

            return (
                block_terminates(stmt.block)
                and all(block_terminates(sec.block) for sec in stmt.elifs)
                and block_terminates(stmt.else_.block)
            )

        case _:
            return False


def check_type(node: AstNode, env: TypeEnv) -> None:
    """Type check the tree rooted at `node`, and annotate expressions in place.

    `ReturnStmt.func` must already be linked,
    as `link_return_statements` in `parser.py` does.

    Raises `CodeError` for a fault in the model,
    positioned at the innermost node that knew where it was,
    and `CompilerError` for anything else.
    """
    try:
        # Children are checked before their parent,
        # so every subexpression carries a type by the time the parent needs it.
        for child in node.children:
            check_type(child, env)

        match node:
            case Ref() as ref:
                match ref.values:
                    case [TickVar() as var] if not var.is_real:
                        raise CodeError(
                            f"{var.name} is the key column of the tick data"
                            " and can't be used in an expression",
                            pos=ref.pos,
                        )

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
                match call.func.value:
                    case BuiltinFunc() as fn:
                        rtype = env.check_func_call(fn.ptypes, fn.rtype, atypes)
                        call.type = rtype
                    case Func() as fn:
                        rtype = "void" if fn.rtype is None else fn.rtype.name
                        ptypes = [param.type.name for param in fn.params]
                        rtype = env.check_func_call(ptypes, rtype, atypes)
                        call.type = rtype
                    case _ as unexpected:
                        raise TypeError(f"{unexpected} is not callable")

            case AssignmentStmt() as stmt:
                obj = stmt.lvalue.value
                match obj:
                    case LocalVariable() | Parameter():
                        ltype = get_type(stmt.lvalue)
                        rtype = get_type(stmt.rvalue)
                        env.check_assign(ltype, rtype)
                    case _:
                        raise TypeError(f"Object {obj} can't be assigned to")

            case UpdateStmt() as stmt:
                obj = stmt.lvalue.value
                match obj:
                    case LocalVariable() | Parameter():
                        ltype = get_type(stmt.lvalue)
                        rtype = get_type(stmt.rvalue)
                        env.check_update(stmt.op, ltype, rtype)
                    case _:
                        raise TypeError(f"Object {obj} can't be updated")

            case ReturnStmt() as stmt:
                assert stmt.func is not None
                if (stmt.arg is None) != (stmt.func.rtype is None):
                    raise TypeError("Return type mismatch")

                if stmt.arg is not None and stmt.func.rtype is not None:
                    atype = get_type(stmt.arg)
                    if not env.is_convertable_to(atype, stmt.func.rtype.name):
                        raise TypeError(
                            f"Return type mismatch: returning {atype} from function of type {stmt.func.rtype.name}"
                        )

            case IfStmt() as stmt:
                cond_type = get_type(stmt.condition)
                if not env.is_numeric(cond_type):
                    raise TypeError("Test expression type not boolean or numeric")

            case ElifSection() as stmt:
                cond_type = get_type(stmt.condition)
                if not env.is_numeric(cond_type):
                    raise TypeError("Condition expression type not boolean or numeric")

            case Func() as func:
                if func.rtype is not None and not block_terminates(func.block):
                    raise TypeError(
                        f"Function {func.name} can complete without returning"
                        f" a value of type {func.rtype.name}:"
                        " every path through the body must return",
                        pos=func.pos,
                    )

            case TickData() as tick_data:
                if not env.is_integral(get_type(tick_data.key_var.type)):
                    raise TypeError(
                        "Key attribute of tick data must be of integral type",
                        tick_data.key_var.type.pos,
                    )

            case PoissonDist() as dist:
                if not env.is_numeric(get_type(dist.mean)):
                    raise TypeError("Expected numeric expression", dist.mean.pos)

            case NormalDist() as dist:
                if not env.is_numeric(get_type(dist.mean)):
                    raise TypeError("Expected numeric expression", dist.mean.pos)

                if not env.is_numeric(get_type(dist.std)):
                    raise TypeError("Expected numeric expression", dist.std.pos)

            case DeterministicDist() as dist:
                if not env.is_numeric(get_type(dist.expr)):
                    raise TypeError("Expected numeric expression", dist.expr.pos)

            case EmberJumpLikelihood():
                if not env.is_numeric(get_type(node.like)):
                    raise TypeError("Expected numeric expression", node.like.pos)

            case EmberDeathProb():
                if not env.is_numeric(get_type(node.prob)):
                    raise TypeError("Expected numeric expression", node.prob.pos)

            case FlameSpreadWeight():
                if not env.is_numeric(get_type(node.weight)):
                    raise TypeError("Expected numeric expression", node.weight.pos)

            case EmberIgnitionProb():
                if not env.is_numeric(get_type(node.prob)):
                    raise TypeError("Expected numeric expression", node.prob.pos)

            case FlameIgnitionProb():
                if not env.is_numeric(get_type(node.prob)):
                    raise TypeError("Expected numeric expression", node.prob.pos)

    # These clauses repeat what `node_error_attributer` in `error.py` does.
    # See "Positions are attributed by the innermost frame that knows one"
    # in docs/developer-notes.md.
    except CodeError as e:
        e.pos = node.pos if e.pos is None else e.pos
        raise e
    except CompilerError:
        raise
    except Exception as e:
        raise CompilerError(f"{node=}") from e
