"""Code generation."""

from .ast_nodes import *
from .builtins import *
from .error import CompilerError
from .templates import render


def mangle(name: str) -> str:
    return "_" + name


# fmt: off
TYPE_TO_CTYPE = {
    "int":   "std::int64_t",
    "uint":  "std::uint64_t",
    "float": "double",
    "bool":  "bool",

    "u8":  "std::uint8_t",
    "u16": "std::uint16_t",
    "u32": "std::uint32_t",
    "u64": "std::uint64_t",

    "i8":  "std::int8_t",
    "i16": "std::int16_t",
    "i32": "std::int32_t",
    "i64": "std::int64_t",

    "f32": "float",
    "f64": "double",

    "str": "std::string",
}

BUILTIN_FN_NAME = {
    "sqrt": "std::sqrt"
}
# fmt: on


def cpp_type(name: str) -> str:
    return TYPE_TO_CTYPE[name]


def cpp_init(name: str) -> str:
    if name == "str":
        return '""'
    else:
        return "0"


def codegen_expr(node: AstNode) -> str:
    match node:
        case Bool() as lit:
            return {True: "true", False: "false"}[lit.value]
        case Int() as lit:
            return str(lit.value)
        case Float() as lit:
            return str(lit.value)
        case Str() as lit:
            return '"' + lit.value + '"'
        case Ref() as ref:
            return mangle(ref.name)
        case UnaryExpr() as expr:
            arg = codegen_expr(expr.arg)
            return f"( {expr.op} ({arg}) )"
        case BinaryExpr() as expr:
            left = codegen_expr(expr.left)
            right = codegen_expr(expr.right)
            if expr.op == "**":
                return f"std::pow( ({left}), ({right}) )"
            else:
                return f"( ({left}) {expr.op} ({right}) )"
        case FuncCall() as call:
            args = [codegen_expr(arg) for arg in call.args]
            args = ", ".join(args)
            match call.func.value():
                case BuiltinFunc() as fn:
                    func = BUILTIN_FN_NAME[fn.name]
                    return f"{func}({args})"
                case _ as unexpected:
                    raise CompilerError(f"unexpected function value {unexpected=}")

    raise CompilerError(f"unexpected node type {node=}")


def codegen_openmp_cpu(node: AstNode) -> str:
    match node:
        case Bool() | Int() | Float() | Str() | Ref() | UnaryExpr() | BinaryExpr():
            return codegen_expr(node)
        case PassStmt() as stmt:
            return "// pass"
        case AssignmentStmt() as stmt:
            lvalue = codegen_expr(stmt.lvalue)
            rvalue = codegen_expr(stmt.rvalue)
            return render("openmp-cpu:assignment_stmt", lvalue=lvalue, rvalue=rvalue)
        case UpdateStmt() as stmt:
            lvalue = codegen_expr(stmt.lvalue)
            rvalue = codegen_expr(stmt.rvalue)
            return render(
                "openmp-cpu:update_stmt", lvalue=lvalue, op=stmt.op, rvalue=rvalue
            )
        case PrintStmt() as stmt:
            args = [codegen_expr(arg) for arg in stmt.args]
            format_string = " ".join(["{}"] * len(args))
            return render("openmp-cpu:print_stmt", format_string=format_string, args=args)
        case ElseSection() as stmt:
            stmts = [codegen_openmp_cpu(stmt) for stmt in stmt.stmts]
            return render("openmp-cpu:else_section", stmts=stmts)
        case ElifSection() as stmt:
            condition = codegen_expr(stmt.condition)
            stmts = [codegen_openmp_cpu(stmt) for stmt in stmt.stmts]
            return render("openmp-cpu:elif_section", condition=condition, stmts=stmts)
        case IfStmt() as stmt:
            condition = codegen_expr(stmt.condition)
            stmts = [codegen_openmp_cpu(stmt) for stmt in stmt.stmts]
            elifs = [codegen_openmp_cpu(section) for section in stmt.elifs]
            else_ = (
                codegen_openmp_cpu(stmt.else_)
                if stmt.else_ is not None
                else "// no else section"
            )
            return render(
                "openmp-cpu:if_stmt",
                condition=condition,
                stmts=stmts,
                elifs=elifs,
                else_=else_,
            )
        case Source() as source:
            lvars = []
            for lvar in source.lvars:
                var = mangle(lvar.name)
                type = cpp_type(lvar.type.name)
                init = cpp_init(lvar.type.name)
                lvars.append((var, type, init))

            stmts = [codegen_openmp_cpu(stmt) for stmt in source.stmts]
            return render("openmp-cpu:main.cpp", stmts=stmts, lvars=lvars)

    raise CompilerError(f"unexpected node type {node=}")
