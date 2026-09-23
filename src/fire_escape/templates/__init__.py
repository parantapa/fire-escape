"""Many-templates-per-file loading for Jinja 2."""

from pathlib import Path
from typing import cast
from dataclasses import dataclass
import importlib.resources

import json5


@dataclass(frozen=True, slots=True)
class TemplateText:
    """One template cut out of a `.jinja` file, ready for Jinja to compile."""

    name: str
    source: str
    filename: str


_TEMPLATE_ANCHOR = __name__

# Process wide cache, keyed by the full `prefix:name`.
# A `.jinja` file is read on each lookup that misses the cache,
# and yields every template it holds,
# so the first lookup of any name in a file populates all of them.
_TEMPLATES: dict[str, TemplateText] = {}


def line_col_from_pos(text: str, loc: int) -> tuple[int, int]:
    """Return the 1 based line and column of an offset into `text`."""
    if not len(text):
        return 1, 1
    sp = text[: loc + 1].splitlines(keepends=True)
    return len(sp), len(sp[-1])


def parse_file(prefix: str, path: Path) -> dict[str, TemplateText]:
    """Split a `.jinja` file into its templates, keyed by `prefix:name`.

    A template starts at a `{#- ... -#}` header.
    The header holds JSON5 object fields without the enclosing braces,
    and `name` is required.
    The template runs to the next header, or to the end of the file.
    Any `{#-` starts a header,
    so a template body cannot hold a Jinja comment opened that way.

    Any exception is annotated with the position of the header being read,
    or with the start of the file for the first header,
    and re-raised.
    """
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
    """Look up `prefix:name`, reading `prefix.jinja` if it is not cached yet.

    Returns the triple that `jinja2.FunctionLoader` expects,
    or None when no such template exists.
    The third element is the uptodate callable,
    and None there means a loaded template is never recompiled.
    """
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
