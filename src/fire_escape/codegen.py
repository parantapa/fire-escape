"""Code generation."""

import jinja2

from .ast_nodes import *
from .builtins import *
from .error import CompilerError
from .templates import load_template

ENVIRONMENT = jinja2.Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.StrictUndefined,
    loader=jinja2.FunctionLoader(load_template),
)


def render(template: str, **kwargs) -> str:
    tpl = ENVIRONMENT.get_template(template)
    return tpl.render(**kwargs)


def mangle(name: str) -> str:
    return "_" + name


ENVIRONMENT.filters["mangle"] = mangle


# fmt: off
TYPE_TO_CTYPE = {
    "int":   "std::int32_t",
    "uint":  "std::uint32_t",
    "float": "float",
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

    "position": "Position",
    "fire_state": "fire_state_t",
}

TYPE_TO_H5TYPE = {
    "int":   "H5::PredType::NATIVE_INT32",
    "uint":  "H5::PredType::NATIVE_UINT32",
    "float": "H5::PredType::NATIVE_FLOAT",
    "bool":  "H5::PredType::NATIVE_UINT8",

    "u8":  "H5::PredType::NATIVE_UINT8",
    "u16": "H5::PredType::NATIVE_UINT16",
    "u32": "H5::PredType::NATIVE_UINT32",
    "u64": "H5::PredType::NATIVE_UINT64",

    "i8":  "H5::PredType::NATIVE_INT8",
    "i16": "H5::PredType::NATIVE_INT16",
    "i32": "H5::PredType::NATIVE_INT32",
    "i64": "H5::PredType::NATIVE_INT64",

    "f32": "H5::PredType::NATIVE_FLOAT",
    "f64": "H5::PredType::NATIVE_DOUBLE",

    "fire_state": "H5::PredType::NATIVE_INT8",
}

TYPE_TO_ARROW_TYPE = {
    "int":   "arrow::int32()",
    "uint":  "arrow::uint32()",
    "float": "arrow::float32()",
    "bool":  "arrow::boolean()",

    "u8":  "arrow::uint8()",
    "u16": "arrow::uint16()",
    "u32": "arrow::uint32()",
    "u64": "arrow::uint64()",

    "i8":  "arrow::int8()",
    "i16": "arrow::int16()",
    "i32": "arrow::int32()",
    "i64": "arrow::int64()",

    "f32": "arrow::float32()",
    "f64": "arrow::float64()",
}

TYPE_TO_ARROW_ARRAY_TYPE = {
    "int":   "arrow::Int32Array",
    "uint":  "arrow::UInt32Array",
    "float": "arrow::FloatArray",
    "bool":  "arrow::BooleanArray",

    "u8":  "arrow::UInt8Array",
    "u16": "arrow::UInt16Array",
    "u32": "arrow::UInt32Array",
    "u64": "arrow::UInt64Array",

    "i8":  "arrow::Int8Array",
    "i16": "arrow::Int16Array",
    "i32": "arrow::Int32Array",
    "i64": "arrow::Int64Array",

    "f32": "arrow::FloatArray",
    "f64": "arrow::DoubleArray",
}

BUILTIN_FN_NAME = {
    "exp": "std::exp",
    "alignment": "alignment",
    "distance": "distance"
}
# fmt: on


def cpp_type(name: str) -> str:
    return TYPE_TO_CTYPE[name]


ENVIRONMENT.filters["cpp_type"] = cpp_type


def h5_type(name: str) -> str:
    return TYPE_TO_H5TYPE[name]


ENVIRONMENT.filters["h5_type"] = h5_type


def arrow_type(name: str) -> str:
    return TYPE_TO_ARROW_TYPE[name]


ENVIRONMENT.filters["arrow_type"] = arrow_type


def arrow_array_type(name: str) -> str:
    return TYPE_TO_ARROW_ARRAY_TYPE[name]


ENVIRONMENT.filters["arrow_array_type"] = arrow_array_type


# Hack required for argparse, which doesn't handle floats well yet.
def cpp_type_config(name: str) -> str:
    if name == "float":
        return "double"
    else:
        return cpp_type(name)


ENVIRONMENT.filters["cpp_type_config"] = cpp_type_config


def cpp_init(name: str) -> str:
    if name == "position":
        return "{0, 0}"
    else:
        return "0"


ENVIRONMENT.filters["cpp_init"] = cpp_init


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
            match ref.values:
                case [LocalVariable() | Parameter() as obj]:
                    return mangle(obj.name)
                case [Config() as obj]:
                    return mangle(obj.name)
                case [TickVar() as obj]:
                    return mangle(obj.name) + "[CUR_TICK]"
                case [Func() as fn]:
                    return mangle(fn.name)
                case [BuiltinFunc() as fn]:
                    return BUILTIN_FN_NAME[fn.name]
                case [BuiltinObject() as obj]:
                    return obj.name.upper()
                case [BuiltinObject() as row, TileVar() as col]:
                    match row.type:
                        case "tile":
                            xindex, yindex, position = "x", "y", "pos"
                        case "src_tile":
                            xindex, yindex, position = "sx", "sy", "src_pos"
                        case "dst_tile":
                            xindex, yindex, position = "dx", "dy", "dst_pos"
                        case _ as unexpected:
                            raise CompilerError(
                                f"unexpected builtin object type {unexpected=}"
                            )

                    if col.type.name == "position":
                        return position
                    else:
                        return mangle(col.name) + f"[{xindex}, {yindex}]"
                case _ as unexpected:
                    raise CompilerError(f"unexpected reference value {unexpected=}")
        case UnaryExpr(op=op, arg=arg):
            arg = codegen_expr(arg)
            return f"( {op} {arg} )"
        case BinaryExpr(left=left, op=op, right=right):
            left = codegen_expr(left)
            right = codegen_expr(right)
            if op == "**":
                return f"std::pow( {left}, {right} )"
            else:
                return f"( {left} {op} {right} )"
        case FuncCall(func=func, args=args):
            args = [codegen_expr(arg) for arg in args]
            args = ", ".join(args)
            match func.value:
                case BuiltinFunc() as fn:
                    func = BUILTIN_FN_NAME[fn.name]
                    return f"{func}({args})"
                case Func() as fn:
                    func = mangle(fn.name)
                    return f"{func}({args})"
                case _ as unexpected:
                    raise CompilerError(f"unexpected function value {unexpected=}")

    raise CompilerError(f"unexpected node type {node=}")


ENVIRONMENT.filters["codegen_expr"] = codegen_expr


def codegen_stmt(node: AstNode) -> str:
    match node:
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

        case ReturnStmt() as stmt:
            if stmt.arg:
                arg = codegen_expr(stmt.arg)
            else:
                arg = None
            return render(
                "openmp-cpu:return_stmt",
                arg=arg,
                pos=stmt.pos,
            )

        case ElseSection() as stmt:
            stmts = [codegen_stmt(stmt) for stmt in stmt.block.stmts]
            return render("openmp-cpu:else_section", stmts=stmts, pos=stmt.pos)

        case ElifSection() as stmt:
            condition = codegen_expr(stmt.condition)
            stmts = [codegen_stmt(stmt) for stmt in stmt.block.stmts]
            return render(
                "openmp-cpu:elif_section",
                condition=condition,
                stmts=stmts,
                pos=stmt.pos,
            )

        case IfStmt() as stmt:
            condition = codegen_expr(stmt.condition)
            stmts = [codegen_stmt(stmt) for stmt in stmt.block.stmts]
            elifs = [codegen_stmt(section) for section in stmt.elifs]
            else_ = (
                codegen_stmt(stmt.else_)
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

    raise CompilerError(f"unexpected node type {node=}")


def codegen_openmp_cpu(node: AstNode | tuple[AstNode, str]) -> str:
    match node:
        case [Func() as fn, "decl"]:
            rtype = "void" if fn.rtype is None else cpp_type(fn.rtype.name)
            name = mangle(fn.name)
            ptypes = [cpp_type(param.type.name) for param in fn.params]
            return render("openmp-cpu:func_decl", name=name, ptypes=ptypes, rtype=rtype)

        case [Func() as fn, "defn"]:
            rtype = "void" if fn.rtype is None else cpp_type(fn.rtype.name)
            name = mangle(fn.name)

            params = [
                (cpp_type(param.type.name), mangle(param.name)) for param in fn.params
            ]
            params = ["%s %s" % p for p in params]

            lvars = []
            for lvar in fn.lvars:
                var = mangle(lvar.name)
                type = cpp_type(lvar.type.name)
                init = cpp_init(lvar.type.name)
                lvars.append((var, type, init))

            stmts = [codegen_stmt(stmt) for stmt in fn.block.stmts]

            return render(
                "openmp-cpu:func_defn",
                name=name,
                params=params,
                rtype=rtype,
                lvars=lvars,
                stmts=stmts,
                pos=fn.pos,
            )

        case Source() as source:
            fn_decls = []
            fn_defns = []

            for fn in source.funcs:
                fn_decls.append(codegen_openmp_cpu((fn, "decl")))
                fn_defns.append(codegen_openmp_cpu((fn, "defn")))

            return render(
                "openmp-cpu:simulator.cpp",
                source=source,
                fn_decls=fn_decls,
                fn_defns=fn_defns,
            )

    raise CompilerError(f"unexpected node type {node=}")
