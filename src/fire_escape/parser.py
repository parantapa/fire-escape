"""Text to a checked AST: parse, build, resolve names, then type check."""

from __future__ import annotations

import importlib.resources
from functools import cache
from collections import ChainMap
from typing import Any, cast

from lark import Lark, Tree, Token
from lark.indenter import Indenter

from .ast_nodes import *
from .ast_nodes import Expression
from .error import *

from .type_check import check_type, TypeEnv
from .builtins import BuiltinObject, add_builtins

GRAMMAR_ANCHOR = __name__
GRAMMAR_FILE = "grammar.lark"


class MyIndenter(Indenter):
    """The postlexer that turns newlines and indentation into tokens."""

    # Lark declares these as abstract properties.
    # A plain class attribute satisfies them at run time,
    # but pyright reports it as an incompatible override.
    NL_type = "_NEWLINE"  # type: ignore
    OPEN_PAREN_types = ["LPAR", "LSQB"]  # type: ignore
    CLOSE_PAREN_types = ["RPAR", "RSQB"]  # type: ignore
    INDENT_type = "_INDENT"  # type: ignore
    DEDENT_type = "_DEDENT"  # type: ignore
    # A tab counts as 8 columns, which is what Python itself assumed
    # before it rejected mixed indentation.
    # This accepts a model that mixes tabs and spaces at one depth,
    # and that model must be rejected.
    tab_len = 8  # type: ignore


@cache
def get_parser() -> Lark:
    """Return the shared LALR parser, building it on the first call."""
    with importlib.resources.path(GRAMMAR_ANCHOR, GRAMMAR_FILE) as path:
        return Lark(
            path.read_text(),
            parser="lalr",
            start="source",
            strict=True,
            propagate_positions=True,
            postlex=MyIndenter(),
        )


def _unary(children, pos: Position) -> Expression:
    """Build a unary expression, or pass the operand through untouched."""
    match children:
        case [op, arg]:
            return UnaryExpr(op=op.value, arg=arg, pos=pos, children=[arg])
        case [arg]:
            return arg
        case _ as unexpected:
            raise CompilerError(f"{unexpected=}")


def _binary_left_assoc(children, pos: Position) -> Expression:
    """Fold a flat `a op b op c` run into a left-leaning tree."""
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


def _is_kept(child: object) -> bool:
    """Return True for a child that carries meaning for the AST."""
    # An optional `[ ]` item absent from the text arrives as None.
    if child is None:
        return False

    # A rule marked with `!` in the grammar keeps every one of its tokens.
    # That overrides the filtering the leading underscore of `_NEWLINE`
    # otherwise applies.
    # The statement terminator is punctuation, so it is dropped here instead.
    return not (isinstance(child, Token) and child.type == "_NEWLINE")


# The return is deliberately left unannotated,
# although the truthful type is `AstNode | str`.
# See "Known rough edges" in docs/developer-notes.md before adding it.
def build_ast(tree: Tree, file: str):
    """Convert a Lark parse tree into the AST, bottom up.

    Returns a bare `str` for an annotation rule such as `tick_var_annot`,
    whose value is the annotation name rather than a node of its own.

    Raises `ParseError` for a duplicate or a missing declaration,
    and `CompilerError` for a tree shape the grammar should not produce.
    """
    children = [
        build_ast(child, file) if isinstance(child, Tree) else child
        for child in tree.children
        if _is_kept(child)
    ]
    pos = Position(file=file, line=tree.meta.line, col=tree.meta.column)

    match tree.data:
        case "bool":
            (child,) = children
            value = child.value == "True"
            return Bool(value=value, pos=pos, children=[])

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

        case (
            "binary_exp"
            | "binary_mul"
            | "binary_add"
            | "binary_cmp"
            | "binary_and"
            | "binary_or"
        ):
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

            # The name is a token, not a node,
            # so the children are rebuilt from the nodes alone.
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

        case "deterministic_dist":
            (child,) = children
            return DeterministicDist(expr=child, pos=pos, children=children)

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

        case "ember_ignition_prob":
            var_name, prob = children
            var_name = var_name.value
            return EmberIgnitionProb(
                var_name=var_name, prob=prob, pos=pos, children=[prob]
            )

        case "create_flames":
            var_name, dist = children
            var_name = var_name.value
            return CreateFlames(var_name=var_name, dist=dist, pos=pos, children=[dist])

        case "flame_spread_weight":
            svar_name, dvar_name, weight = children
            svar_name = svar_name.value
            dvar_name = dvar_name.value
            return FlameSpreadWeight(
                svar_name=svar_name,
                dvar_name=dvar_name,
                weight=weight,
                pos=pos,
                children=[weight],
            )

        case "flame_ignition_prob":
            var_name, prob = children
            var_name = var_name.value
            return FlameIgnitionProb(
                var_name=var_name, prob=prob, pos=pos, children=[prob]
            )

        case "burn_time":
            var_name, dist = children
            var_name = var_name.value
            return BurnTime(var_name=var_name, dist=dist, pos=pos, children=[dist])

        case "fire_model":
            create_embers = None
            ember_jump_likelihood = None
            ember_death_prob = None
            ember_ignition_prob = None
            create_flames = None
            flame_spread_weight = None
            flame_ignition_prob = None
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
                    case EmberIgnitionProb():
                        if ember_ignition_prob is not None:
                            raise ParseError(
                                "ember-ignition-prob has been defined multiple times",
                                pos=child.pos,
                            )
                        ember_ignition_prob = child
                    case CreateFlames():
                        if create_flames is not None:
                            raise ParseError(
                                "create-flames has been defined multiple times",
                                pos=child.pos,
                            )
                        create_flames = child
                    case FlameSpreadWeight():
                        if flame_spread_weight is not None:
                            raise ParseError(
                                "flame-spread-weight has been defined multiple times",
                                pos=child.pos,
                            )
                        flame_spread_weight = child
                    case FlameIgnitionProb():
                        if flame_ignition_prob is not None:
                            raise ParseError(
                                "flame-ignition-prob has been defined multiple times",
                                pos=child.pos,
                            )
                        flame_ignition_prob = child
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
            if ember_ignition_prob is None:
                raise ParseError(
                    "ember-ignition-prob has not been defined",
                    pos=pos,
                )
            if create_flames is None:
                raise ParseError(
                    "create-flames has not been defined",
                    pos=pos,
                )
            if flame_spread_weight is None:
                raise ParseError("flame-spread-weight has not been defined", pos=pos)
            if flame_ignition_prob is None:
                raise ParseError(
                    "flame-ignition-prob has not been defined",
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
                ember_ignition_prob=ember_ignition_prob,
                create_flames=create_flames,
                flame_spread_weight=flame_spread_weight,
                flame_ignition_prob=flame_ignition_prob,
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
def build_scope(node: AstNode, scope: ChainMap[str, Any]) -> None:
    """Attach a scope to every node that resolves names, and bind each declaration.

    Mutates the tree and `scope` in place.
    A function, and each fire-model clause, opens a child scope,
    so a tile variable bound by a clause does not leak out of it.

    Raises `ReferenceError` when a name is defined twice in one scope.
    """
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

        case (
            CreateEmbers()
            | EmberIgnitionProb()
            | CreateFlames()
            | FlameIgnitionProb()
            | BurnTime() as obj
        ):
            scope = scope.new_child()

            obj.scope = scope
            obj.scope[obj.var_name] = BuiltinObject(obj.var_name, "tile")

        case EmberJumpLikelihood() | EmberDeathProb() | FlameSpreadWeight() as obj:
            scope = scope.new_child()

            obj.scope = scope
            obj.scope[obj.svar_name] = BuiltinObject(obj.svar_name, "src_tile")
            obj.scope[obj.dvar_name] = BuiltinObject(obj.dvar_name, "dst_tile")

    for child in node.children:
        build_scope(child, scope)


@node_error_attributer
def collect_local_variables(node: AstNode) -> None:
    """Fill each function's `lvars` from the variables its own scope holds."""
    match node:
        case Func() as func:
            assert func.scope is not None

            for var in func.scope.maps[0].values():
                if isinstance(var, LocalVariable):
                    func.lvars.append(var)

    for child in node.children:
        collect_local_variables(child)


@node_error_attributer
def link_return_statements(node: AstNode, func: Func | None) -> None:
    """Point each return statement at its enclosing function, and back.

    `func` is the function the walk is currently inside, and None at the top.
    """
    match node:
        case Func() as func:
            # `as func` rebinds the parameter for the recursion below,
            # which is how a return statement in the body reaches this function.
            # Without the capture, `func` stays None,
            # and the assert on the return statement fails.
            pass
        case ReturnStmt() as stmt:
            assert func is not None
            assert stmt.func is None
            stmt.func = func
            func.return_stmts.append(stmt)

    for child in node.children:
        link_return_statements(child, func)


@node_error_attributer
def populate_tile_objects(node: AstNode, tile_data: TileData) -> None:
    """Give each tile variable bound by a fire-model clause its attributes.

    Expects `build_scope` to have run on the same tree first.
    """
    match node:
        case (
            CreateEmbers()
            | EmberIgnitionProb()
            | CreateFlames()
            | FlameIgnitionProb()
            | BurnTime() as obj
        ):
            assert obj.scope is not None
            ref: BuiltinObject = obj.scope[obj.var_name]

            for tile_var in tile_data.tile_vars:
                ref.attrs[tile_var.name] = tile_var

        case EmberJumpLikelihood() | EmberDeathProb() | FlameSpreadWeight() as obj:
            assert obj.scope is not None
            for var_name in [obj.svar_name, obj.dvar_name]:
                ref: BuiltinObject = obj.scope[var_name]
                for tile_var in tile_data.tile_vars:
                    ref.attrs[tile_var.name] = tile_var

    for child in node.children:
        populate_tile_objects(child, tile_data)


def populate_source_opts(source: Source) -> None:
    """Fill `source.opts` with every option, taking the default where none is declared.

    Raises `CodeError` for an unknown option, for one declared twice,
    or for a chunk size smaller than the matching maximum jump.
    """
    source.opts["max_jump_x"] = 1
    source.opts["max_jump_y"] = 1
    source.opts["max_ember_count"] = 100
    # chunk_size_x and chunk_size_y are validated below
    # and emitted as constants in the generated C++,
    # but nothing there reads them, so setting one has no effect.
    # They are here for a tiled traversal that is not written yet.
    source.opts["chunk_size_x"] = 64
    source.opts["chunk_size_y"] = 64

    seen: set[str] = set()
    for opt in source.options:
        if opt.name not in source.opts:
            raise CodeError(f"Unknown option {opt.name}", pos=opt.pos)
        if opt.name in seen:
            raise CodeError(f"Option {opt.name} has already been defined", pos=opt.pos)
        seen.add(opt.name)
        source.opts[opt.name] = opt.value

    if source.opts["chunk_size_x"] < source.opts["max_jump_x"]:
        raise CodeError(
            f"chunk_size_x ({source.opts["chunk_size_x"]}) is less than max_jump_x ({source.opts["max_jump_x"]})",
            pos=source.pos,
        )

    if source.opts["chunk_size_y"] < source.opts["max_jump_y"]:
        raise CodeError(
            f"chunk_size_y ({source.opts["chunk_size_y"]}) is less than max_jump_y ({source.opts["max_jump_y"]})",
            pos=source.pos,
        )


def validate_tick_data(tick_data: TickData) -> None:
    """Raise `CodeError` unless exactly one tick variable is annotated `key`."""
    num_keys = sum(1 for var in tick_data.tick_vars if "key" in var.annots)
    if num_keys != 1:
        raise CodeError(
            "One and only one attribute of tick data annotated as 'key' attribute",
            tick_data.pos,
        )


def validate_tile_data(tile_data: TileData) -> None:
    """Check that the tile data has one `position` and one `fire_state` variable.

    Raises `CodeError` unless exactly one tile variable has each of those types.
    """
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


def parse(file: str, text: str) -> Source:
    """Parse and fully check FFSL text, and return a tree ready for a backend.

    `file` is recorded in every node's position,
    which the error messages and the generated `#line` directives both use.

    Raises `CodeError` for a fault the checks find in the model,
    Lark's `UnexpectedInput` or `DedentError` for text the grammar rejects,
    and `CompilerError` for a fault in the compiler.
    """
    parser = get_parser()

    tree = parser.parse(text)
    source: Source = cast(Source, build_ast(tree, file))

    # The order of these passes is load bearing.
    # See docs/developer-notes.md.
    root_scope = ChainMap()
    add_builtins(root_scope)
    build_scope(source, root_scope)
    assert source.scope is not None

    collect_local_variables(source)
    link_return_statements(source, None)
    populate_tile_objects(source, source.tile_data)
    populate_source_opts(source)
    validate_tick_data(source.tick_data)
    validate_tile_data(source.tile_data)

    env = TypeEnv.new()
    check_type(source, env)
    return source
