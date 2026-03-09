"""Parser module."""

from __future__ import annotations

import importlib.resources
from functools import cache
from collections import ChainMap
from typing import Any, cast

from lark import Lark, Tree, Token
from lark.indenter import Indenter

from .ast_nodes import *
from .error import *

from .type_check import check_type, TypeEnv
from .builtins import BuiltinObject, add_builtins

GRAMMAR_ANCHOR = __name__
GRAMMAR_FILE = "grammar.lark"


class MyIndenter(Indenter):
    NL_type = "_NEWLINE"  # type: ignore
    OPEN_PAREN_types = ["LPAR", "LSQB"]  # type: ignore
    CLOSE_PAREN_types = ["RPAR", "RSQB"]  # type: ignore
    INDENT_type = "_INDENT"  # type: ignore
    DEDENT_type = "_DEDENT"  # type: ignore
    tab_len = 8  # type: ignore


@cache
def get_parser() -> Lark:
    with importlib.resources.path(GRAMMAR_ANCHOR, GRAMMAR_FILE) as path:
        return Lark(
            path.read_text(),
            parser="lalr",
            start="source",
            strict=True,
            propagate_positions=True,
            postlex=MyIndenter(),
        )


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
        case "bool":
            (child,) = children
            value = child.value == "True"
            return Bool(value=True, pos=pos, children=[])

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
            name = tuple(child.value for child in children)
            if len(name) == 1:
                name = name[0]
            return Ref(name=name, pos=pos, children=[])

        case "unary_neg" | "unary_not":
            return _unary(children, pos)

        case "binary_exp" | "binary_mul" | "binary_add" | "binary_cmp" | "binary_and":
            return _binary_left_assoc(children, pos)

        case "func_call":
            func, *args = children
            return FuncCall(func=func, args=args, pos=pos, children=children)

        case "type":
            match children:
                case [name]:
                    return TypeRef(
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
                case [lvalue, typ, rvalue]:
                    var = LocalVariable(
                        name=lvalue.name,
                        type=typ,
                        pos=pos,
                        children=[typ],
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

        case "return_stmt":
            if children:
                return ReturnStmt(arg=children[0], pos=pos, children=children)
            else:
                return ReturnStmt(arg=None, pos=pos, children=[])

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

        case "param":
            name, typ = children
            return Parameter(name=name.value, type=typ, pos=pos, children=[typ])

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

        case "option":
            name, value = children
            name = name.value
            value = int(value.value)
            return Option(name=name, value=value, pos=pos, children=[])

        case "config":
            name, type, default = children
            name = name.value
            return Config(
                name=name, type=type, default=default, pos=pos, children=[type, default]
            )

        case "tick_var_annot" | "tile_var_annot":
            (child,) = children
            return child.value

        case "tick_var":
            name, type, *annots = children
            name = name.value
            return TickVar(
                name=name, type=type, annots=annots, pos=pos, children=[type]
            )

        case "tick_data":
            return TickData(tick_vars=children, pos=pos, children=children)

        case "tile_var":
            name, type, *annots = children
            name = name.value
            return TileVar(
                name=name, type=type, annots=annots, pos=pos, children=[type]
            )

        case "tile_data":
            return TileData(tile_vars=children, pos=pos, children=children)

        case "poisson_dist":
            (child,) = children
            return PoissonDist(mean=child, pos=pos, children=children)

        case "normal_dist":
            mean, std = children
            return NormalDist(mean=mean, std=std, pos=pos, children=children)

        case "create_embers":
            var_name, dist = children
            var_name = var_name.value
            return CreateEmbers(var_name=var_name, dist=dist, pos=pos, children=[dist])

        case "ember_jump_likelihood":
            svar_name, dvar_name, like = children
            svar_name = svar_name.value
            dvar_name = dvar_name.value
            return EmberJumpLikelihood(
                svar_name=svar_name,
                dvar_name=dvar_name,
                like=like,
                pos=pos,
                children=[like],
            )

        case "ember_death_prob":
            svar_name, dvar_name, prob = children
            svar_name = svar_name.value
            dvar_name = dvar_name.value
            return EmberDeathProb(
                svar_name=svar_name,
                dvar_name=dvar_name,
                prob=prob,
                pos=pos,
                children=[prob],
            )

        case "ignition_prob":
            var_name, prob = children
            var_name = var_name.value
            return IgnitionProb(var_name=var_name, prob=prob, pos=pos, children=[prob])

        case "burn_time":
            var_name, dist = children
            var_name = var_name.value
            return BurnTime(var_name=var_name, dist=dist, pos=pos, children=[dist])

        case "fire_model":
            create_embers = None
            ember_jump_likelihood = None
            ember_death_prob = None
            ignition_prob = None
            burn_time = None

            for child in children:
                match child:
                    case CreateEmbers():
                        if create_embers is not None:
                            raise ParseError(
                                "create-embers has been defined multiple times",
                                pos=child.pos,
                            )
                        create_embers = child
                    case EmberJumpLikelihood():
                        if ember_jump_likelihood is not None:
                            raise ParseError(
                                "ember-jump-likelihood has been defined multiple times",
                                pos=child.pos,
                            )
                        ember_jump_likelihood = child
                    case EmberDeathProb():
                        if ember_death_prob is not None:
                            raise ParseError(
                                "ember-death-prob has been defined multiple times",
                                pos=child.pos,
                            )
                        ember_death_prob = child
                    case IgnitionProb():
                        if ignition_prob is not None:
                            raise ParseError(
                                "ignition-prob has been defined multiple times",
                                pos=child.pos,
                            )
                        ignition_prob = child
                    case BurnTime():
                        if burn_time is not None:
                            raise ParseError(
                                "burn-time has been defined multiple times",
                                pos=child.pos,
                            )
                        burn_time = child
                    case _ as unexpected:
                        raise CompilerError(f"{unexpected=}")

            if create_embers is None:
                raise ParseError(
                    "create-embers has not been defined",
                    pos=pos,
                )
            if ember_jump_likelihood is None:
                raise ParseError(
                    "ember-jump-likelihood has not been defined",
                    pos=pos,
                )
            if ember_death_prob is None:
                raise ParseError(
                    "ember-death-prob has not been defined",
                    pos=pos,
                )
            if ignition_prob is None:
                raise ParseError(
                    "ignition-prob has not been defined",
                    pos=pos,
                )
            if burn_time is None:
                raise ParseError(
                    "burn-time has not been defined",
                    pos=pos,
                )

            return FireModel(
                create_embers=create_embers,
                ember_jump_likelihood=ember_jump_likelihood,
                ember_death_prob=ember_death_prob,
                ignition_prob=ignition_prob,
                burn_time=burn_time,
                pos=pos,
                children=children,
            )

        case "source":
            options = []
            configs = []
            funcs = []
            tick_data = None
            tile_data = None
            fire_model = None
            for child in children:
                match child:
                    case Option():
                        options.append(child)
                    case Config():
                        configs.append(child)
                    case Func():
                        funcs.append(child)
                    case TickData():
                        if tick_data is not None:
                            raise ParseError(
                                "Tick data has been defined multiple times",
                                pos=child.pos,
                            )
                        tick_data = child
                    case TileData():
                        if tile_data is not None:
                            raise ParseError(
                                "Tile data has been defined multiple times",
                                pos=child.pos,
                            )
                        tile_data = child
                    case FireModel():
                        if fire_model is not None:
                            raise ParseError(
                                "Fire model has been defined multiple times",
                                pos=child.pos,
                            )
                        fire_model = child
                    case _ as unexpected:
                        raise CompilerError(f"{unexpected=}")

            if tick_data is None:
                raise ParseError("Tick data has not been defined", pos=pos)
            if tile_data is None:
                raise ParseError("Tile data has not been defined", pos=pos)
            if fire_model is None:
                raise ParseError("Fire model has not been defined", pos=pos)

            return Source(
                options=options,
                configs=configs,
                funcs=funcs,
                tick_data=tick_data,
                tile_data=tile_data,
                fire_model=fire_model,
                pos=pos,
                children=children,
            )

        case _ as unexpected:
            raise CompilerError(f"unexpected tree.data={unexpected}; {children=}")


@node_error_attributer
def build_scope(node: AstNode, scope: ChainMap[str, Any]):
    match node:
        case Source() as source:
            source.scope = scope

        case Func() as func:
            if func.name in scope.maps[0]:
                raise ReferenceError(
                    f"{func.name} has already been defined.",
                    pos=func.pos,
                )
            scope[func.name] = func

            scope = scope.new_child()
            func.scope = scope

        case Ref() as ref:
            ref.scope = scope

        case LocalVariable() | Parameter() as obj:
            if obj.name in scope.maps[0]:
                raise ReferenceError(
                    f"{obj.name} has already been defined.",
                    pos=obj.pos,
                )
            scope[obj.name] = obj

        case Config() as config:
            if config.name in scope.maps[0]:
                raise ReferenceError(
                    f"{config.name} has already been defined.",
                    pos=config.pos,
                )
            scope[config.name] = config

        case TickVar() as var:
            if var.name in scope.maps[0]:
                raise ReferenceError(
                    f"{var.name} has already been defined.",
                    pos=var.pos,
                )
            scope[var.name] = var

        case CreateEmbers() | IgnitionProb() | BurnTime() as obj:
            scope = scope.new_child()

            obj.scope = scope
            obj.scope[obj.var_name] = BuiltinObject(obj.var_name, "tile")

        case EmberJumpLikelihood() | EmberDeathProb() as obj:
            scope = scope.new_child()

            obj.scope = scope
            obj.scope[obj.svar_name] = BuiltinObject(obj.svar_name, "src_tile")
            obj.scope[obj.dvar_name] = BuiltinObject(obj.dvar_name, "dst_tile")

    for child in node.children:
        build_scope(child, scope)


@node_error_attributer
def collect_local_varaibles(node: AstNode):
    match node:
        case Func() as func:
            assert func.scope is not None

            for var in func.scope.maps[0].values():
                if isinstance(var, LocalVariable):
                    func.lvars.append(var)

    for child in node.children:
        collect_local_varaibles(child)


@node_error_attributer
def link_return_statements(node: AstNode, func: Func | None):
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


@node_error_attributer
def populate_tile_objects(node: AstNode, tile_data: TileData):
    match node:
        case CreateEmbers() | IgnitionProb() | BurnTime() as obj:
            assert obj.scope is not None
            ref: BuiltinObject = obj.scope[obj.var_name]

            for tile_var in tile_data.tile_vars:
                ref.attrs[tile_var.name] = tile_var

        case EmberJumpLikelihood() | EmberDeathProb() as obj:
            assert obj.scope is not None
            for var_name in [obj.svar_name, obj.dvar_name]:
                ref: BuiltinObject = obj.scope[var_name]
                for tile_var in tile_data.tile_vars:
                    ref.attrs[tile_var.name] = tile_var

    for child in node.children:
        populate_tile_objects(child, tile_data)


def populate_source_opts(source: Source):
    source.opts["max_jump_x"] = 1
    source.opts["max_jump_y"] = 1
    source.opts["max_ember_count"] = 100

    for opt in source.options:
        if opt.name not in source.opts:
            raise CodeError(f"Unknown option {opt.name}", pos=opt.pos)
        source.opts[opt.name] = opt.value


def validate_tick_data(tick_data: TickData):
    num_keys = sum(1 for var in tick_data.tick_vars if "key" in var.annots)
    if num_keys != 1:
        raise CodeError(
            "One and only one attribute of tick data annotated as 'key' attribute",
            tick_data.pos,
        )


def validate_tile_data(tile_data: TileData):
    num_positions = sum(1 for var in tile_data.tile_vars if var.type.name == "position")
    if num_positions != 1:
        raise CodeError(
            "One and only one attribute of tile data must have type 'position'",
            tile_data.pos,
        )

    num_states = sum(1 for var in tile_data.tile_vars if var.type.name == "fire_state")
    if num_states != 1:
        raise CodeError(
            "One and only one attribute of tile data must have type 'fire_state'",
            tile_data.pos,
        )


def parse(file: str, text: str):
    parser = get_parser()

    tree = parser.parse(text)
    source: Source = cast(Source, build_ast(tree, file))

    root_scope = ChainMap()
    add_builtins(root_scope)
    build_scope(source, root_scope)
    assert source.scope is not None

    collect_local_varaibles(source)
    link_return_statements(source, None)
    populate_tile_objects(source, source.tile_data)
    populate_source_opts(source)
    validate_tick_data(source.tick_data)
    validate_tile_data(source.tile_data)

    env = TypeEnv.new()
    check_type(source, env)
    return source
