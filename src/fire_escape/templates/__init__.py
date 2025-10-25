"""Jinja 2 Template Utilities."""

from pathlib import Path
from typing import cast, overload, Literal
from dataclasses import dataclass
import importlib.resources

import json5
import jinja2

from ..error import Position


@dataclass(frozen=True, slots=True)
class TemplateText:
    name: str
    source: str
    filename: str


_TEMPLATE_ANCHOR = __name__
_TEMPLATES: dict[str, TemplateText] = {}


def line_col_from_pos(text: str, loc: int) -> tuple[int, int]:
    if not len(text):
        return 1, 1
    sp = text[: loc + 1].splitlines(keepends=True)
    return len(sp), len(sp[-1])


def parse_file(prefix: str, path: Path) -> dict[str, TemplateText]:
    ret: dict[str, TemplateText] = {}

    text = path.read_text()
    pos = 0

    while True:
        line, col = line_col_from_pos(text, pos)
        head_start = text.find("{#-", pos)
        if head_start == -1:
            return ret

        try:
            head_end = text.find("-#}", head_start)
            if head_end == -1:
                raise ValueError("Unable to find end of header")

            body_end = text.find("{#-", head_end)
            if body_end == -1:
                body_end = len(text)
            pos = body_end

            header = text[head_start + 3 : head_end]
            header = "{" + header + "}"
            header = json5.loads(header)
            header = cast(dict, header)

            name = prefix + ":" + header["name"]

            source = text[head_end + 3 : body_end].strip()

            template_text = TemplateText(name, source, str(path))
            ret[name] = template_text
        except Exception as e:
            e.add_note("Failed to parse template file")
            e.add_note(f"Position: {path}:{line}:{col}")
            raise e


def load_template(name: str) -> tuple[str, str, None] | None:
    if name in _TEMPLATES:
        tpl = _TEMPLATES[name]
        return tpl.source, tpl.filename, None

    prefix = name.split(":")[0]
    filename = prefix + ".jinja"

    with importlib.resources.path(_TEMPLATE_ANCHOR, filename) as path:
        if path.exists():
            tpls = parse_file(prefix, path)
            _TEMPLATES.update(tpls)

    if name in _TEMPLATES:
        tpl = _TEMPLATES[name]
        return tpl.source, tpl.filename, None

    return None


_ENVIRONMENT = jinja2.Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.StrictUndefined,
    loader=jinja2.FunctionLoader(load_template),
)


@overload
def render(
    template: Literal["openmp-cpu:assignment_stmt"],
    *,
    lvalue: str,
    rvalue: str,
    pos: Position,
) -> str: ...


@overload
def render(
    template: Literal["openmp-cpu:update_stmt"],
    *,
    lvalue: str,
    op: str,
    rvalue: str,
    pos: Position,
) -> str: ...


@overload
def render(
    template: Literal["openmp-cpu:print_stmt"],
    *,
    format_string: str,
    args: list[str],
    pos: Position,
) -> str: ...


@overload
def render(
    template: Literal["openmp-cpu:else_section"], *, stmts: list[str], pos: Position
) -> str: ...


@overload
def render(
    template: Literal["openmp-cpu:elif_section"],
    *,
    condition: str,
    stmts: list[str],
    pos: Position,
) -> str: ...


@overload
def render(
    template: Literal["openmp-cpu:if_stmt"],
    *,
    condition: str,
    stmts: list[str],
    elifs: list[str],
    else_: str,
    pos: Position,
) -> str: ...


@overload
def render(
    template: Literal["openmp-cpu:main.cpp"],
    *,
    stmts: list[str],
    lvars: list[tuple[str, str, str]],
) -> str: ...


@overload
def render(template: Literal["openmp-cpu:CMakeLists.txt"], *, module: str) -> str: ...


def render(template, **kwargs) -> str:
    tpl = _ENVIRONMENT.get_template(template)
    return tpl.render(**kwargs)
