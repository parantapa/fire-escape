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


_JSON_EXPR_TEMPLATE = """
[](){
    try {
        return %(expr)s;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(
            fmt::format(
                "bad json expression:{}:{}:{}: {}",
                "%(file)s", %(line)s, %(col)s, e.what()
            )
        );
    }
}()
"""
_JSON_EXPR_TEMPLATE = " ".join(_JSON_EXPR_TEMPLATE.split())


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
        case UnaryExpr(op=op, arg=arg):
            arg = codegen_expr(arg)
            return f"( {op} ({arg}) )"
        case BinaryExpr(left=left, op=op, right=right):
            left = codegen_expr(left)
            right = codegen_expr(right)
            if op == "**":
                return f"std::pow( ({left}), ({right}) )"
            else:
                return f"( ({left}) {op} ({right}) )"
        case FuncCall(func=func, args=args):
            args = [codegen_expr(arg) for arg in args]
            args = ", ".join(args)
            match func.value():
                case BuiltinFunc() as fn:
                    func = BUILTIN_FN_NAME[fn.name]
                    return f"{func}({args})"
                case _ as unexpected:
                    raise CompilerError(f"unexpected function value {unexpected=}")
        case JsonExpr(jvar=jvar, idxs=idxs, type=rtype, pos=pos):
            idxs = [codegen_expr(idx) for idx in idxs]
            idxs = [f"[{idx}]" for idx in idxs]
            idxs = "".join(idxs)

            rtype = cpp_type(rtype.name)

            match jvar.value():
                case BuiltinConst(name=cname):
                    expr_str = f"{cname}{idxs}.template get<{rtype}>()"
                    return _JSON_EXPR_TEMPLATE % dict(
                        expr=expr_str, file=pos.file, line=pos.line, col=pos.col
                    )
                case _ as unexpected:
                    raise CompilerError(f"unexpected json variable {unexpected=}")

    raise CompilerError(f"unexpected node type {node=}")


def codegen_openmp_cpu(node: AstNode) -> str:
    match node:
        case Bool() | Int() | Float() | Str() | Ref() | UnaryExpr() | BinaryExpr() | JsonExpr():
            return codegen_expr(node)
        case PassStmt() as stmt:
            return "// pass"
        case AssignmentStmt() as stmt:
            lvalue = codegen_expr(stmt.lvalue)
            rvalue = codegen_expr(stmt.rvalue)
            return render(
                "openmp-cpu:assignment_stmt", lvalue=lvalue, rvalue=rvalue, pos=stmt.pos
            )
        case UpdateStmt() as stmt:
            lvalue = codegen_expr(stmt.lvalue)
            rvalue = codegen_expr(stmt.rvalue)
            return render(
                "openmp-cpu:update_stmt",
                lvalue=lvalue,
                op=stmt.op,
                rvalue=rvalue,
                pos=stmt.pos,
            )
        case PrintStmt() as stmt:
            args = [codegen_expr(arg) for arg in stmt.args]
            format_string = " ".join(["{}"] * len(args))
            return render(
                "openmp-cpu:print_stmt",
                format_string=format_string,
                args=args,
                pos=stmt.pos,
            )
        case ElseSection() as stmt:
            stmts = [codegen_openmp_cpu(stmt) for stmt in stmt.stmts]
            return render("openmp-cpu:else_section", stmts=stmts, pos=stmt.pos)
        case ElifSection() as stmt:
            condition = codegen_expr(stmt.condition)
            stmts = [codegen_openmp_cpu(stmt) for stmt in stmt.stmts]
            return render(
                "openmp-cpu:elif_section",
                condition=condition,
                stmts=stmts,
                pos=stmt.pos,
            )
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
                pos=stmt.pos,
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
