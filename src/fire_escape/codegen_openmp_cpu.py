"""The OpenMP CPU backend: an FFSL tree in, a buildable C++ project out."""

import math
from typing import Any
from pathlib import Path

import click
import jinja2

from .parser import parse
from .type_check import get_type
from .ast_nodes import *
from .builtins import *
from .error import CodeError, CompilerError, Position
from .templates import load_template

ENVIRONMENT = jinja2.Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.StrictUndefined,
    loader=jinja2.FunctionLoader(load_template),
    extensions=["jinja2.ext.debug", "jinja2.ext.do"],
)


def render(template: str, **kwargs: Any) -> str:
    """Render a named template against the given context."""
    tpl = ENVIRONMENT.get_template(template)
    return tpl.render(**kwargs)


def mangle(name: str) -> str:
    """Return the C++ identifier standing for an FFSL name."""
    # Every name from the model is prefixed.
    # No model can then collide with an identifier the template
    # already uses, such as `x` or `pos`.
    # A leading underscore is reserved at file scope in C++,
    # and the model's functions, configs and data arrays
    # are declared at file scope.
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

BUILTIN_FN_NAME = {
    "exp": "std::expf",
    "alignment": "alignment",
    "distance": "distance"
}
# fmt: on


# C++ types that already divide in floating point.
FLOAT_CTYPES = frozenset(["float", "double"])

# Largest finite magnitude of an IEEE 754 binary32 value.
FLT_MAX = 3.4028234663852886e38


def cpp_type(name: str) -> str:
    """Return the C++ type standing for an FFSL type name."""
    return TYPE_TO_CTYPE[name]


ENVIRONMENT.filters["cpp_type"] = cpp_type


def h5_type(name: str) -> str:
    """Return the HDF5 predefined type standing for an FFSL type name."""
    return TYPE_TO_H5TYPE[name]


ENVIRONMENT.filters["h5_type"] = h5_type


def cpp_type_config(name: str) -> str:
    """Return the C++ type a config flag of this FFSL type parses into.

    This is `cpp_type` except for `float`, which widens to `double`,
    so a caller comparing the two decides whether a cast is needed.
    """
    # See "Config values of type `float` are stored as `double`"
    # in docs/developer-notes.md.
    if name == "float":
        return "double"
    else:
        return cpp_type(name)


ENVIRONMENT.filters["cpp_type_config"] = cpp_type_config


def cpp_init(name: str) -> str:
    """Return the C++ initializer that zeroes a variable of this FFSL type."""
    if name == "position":
        return "{0, 0}"
    else:
        return "0"


ENVIRONMENT.filters["cpp_init"] = cpp_init


def cpp_float_literal(value: float, pos: Position) -> str:
    """Render a Python float as a C++ `float` literal.

    Raises `CodeError` when the value is not finite,
    or when its magnitude is out of range for a binary32 float.
    """
    # An unsuffixed C++ floating literal has type double, which pulls
    # the surrounding expression into double precision.
    # Python's repr always produces a decimal point or an exponent,
    # so an `f` suffix is enough to make the literal well formed.
    if not math.isfinite(value):
        raise CodeError(f"Float literal {value!r} is not a finite number", pos=pos)

    if abs(value) > FLT_MAX:
        raise CodeError(
            f"Float literal {value!r} is out of range for a 32 bit float",
            pos=pos,
        )

    text = repr(value)
    assert "." in text or "e" in text or "E" in text
    return text + "f"


def codegen_expr(node: AstNode) -> str:
    """Return the C++ expression for an already type-checked expression node.

    Raises `CodeError` for a float literal that `cpp_float_literal` rejects.
    Raises `CompilerError` for a node the backend does not handle,
    which is a fault in the compiler rather than in the model.
    """
    match node:
        case Bool() as lit:
            return {True: "true", False: "false"}[lit.value]
        case Int() as lit:
            return str(lit.value)
        case Float() as lit:
            return cpp_float_literal(lit.value, lit.pos)
        case Str() as lit:
            return '"' + lit.value + '"'
        case Ref() as ref:
            match ref.values:
                case [LocalVariable() | Parameter() as obj]:
                    return mangle(obj.name)
                case [Config() as obj]:
                    # The narrowing half of `cpp_type_config`.
                    # See "Config values of type `float` are stored as `double`"
                    # in docs/developer-notes.md.
                    ctype = cpp_type(obj.type.name)
                    if ctype == cpp_type_config(obj.type.name):
                        return mangle(obj.name)
                    return f"static_cast<{ctype}>({mangle(obj.name)})"
                case [TickVar() as obj]:
                    return mangle(obj.name) + "[CUR_TICK]"
                case [Func() as fn]:
                    return mangle(fn.name)
                case [BuiltinFunc() as fn]:
                    return BUILTIN_FN_NAME[fn.name]
                case [BuiltinObject() as obj]:
                    return obj.name.upper()
                case [BuiltinObject() as row, TileVar() as col]:
                    # These are the tile coordinates and the `Position` locals
                    # that simulator.cpp declares around the model expressions.
                    # init_change_time is the exception, and declares no `pos`.
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
        case BinaryExpr(left=lexpr, op=op, right=rexpr):
            left = codegen_expr(lexpr)
            right = codegen_expr(rexpr)
            if op == "**":
                return f"std::powf( {left}, {right} )"
            elif op == "/":
                # `/` is always true division.
                # See "Division is always true division" in docs/developer-notes.md.
                if cpp_type(get_type(lexpr)) not in FLOAT_CTYPES:
                    left = f"float( {left} )"
                return f"( {left} / {right} )"
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
    """Return the C++ statement for a statement node.

    Raises `CompilerError` for a node the backend does not handle.
    """
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


def codegen(node: AstNode | tuple[AstNode, str]) -> str:
    """Return the C++ text for a source tree, or for one part of a function.

    A `Func` is paired with `"decl"` or `"defn"` to pick which of the two
    it renders as. A `Source` renders the whole translation unit.

    Raises `CompilerError` for anything else.
    """
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
                fn_decls.append(codegen((fn, "decl")))
                fn_defns.append(codegen((fn, "defn")))

            lines = render(
                "openmp-cpu:simulator.cpp",
                source=source,
                fn_decls=fn_decls,
                fn_defns=fn_defns,
            ).split("\n")

            new_lines = []
            for i, line in enumerate(lines, 1):
                if line.strip() == "#endline":
                    # See the developer notes on `#endline`.
                    # `#line N` numbers the line after the directive,
                    # and the directive itself sits on line i.
                    line = line.replace("#endline", f'#line {i + 1} "simulator.cpp"')
                new_lines.append(line)

            return "\n".join(new_lines)

    raise CompilerError(f"unexpected node type {node=}")


@click.command(name="compile")
@click.option(
    "-i",
    "--input-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="FFSL simulator code.",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="C++ project directory.",
)
def compile_cmd(input_file: Path, output_dir: Path) -> None:
    """Compile the FFSL code to a C++ project."""
    output_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
    source = parse(str(input_file), input_file.read_text())

    with open(output_dir / "simulator.cpp", "wt") as fobj:
        code = codegen(source)
        fobj.write(code)

    with open(output_dir / "CMakeLists.txt", "wt") as fobj:
        code = render("openmp-cpu:CMakeLists.txt")
        fobj.write(code)

    with open(output_dir / "conanfile.py", "wt") as fobj:
        code = render("openmp-cpu:conanfile.py")
        fobj.write(code)
