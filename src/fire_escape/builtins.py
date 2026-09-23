"""Names every FFSL model can reach without declaring them."""

from __future__ import annotations

__all__ = ["BuiltinFunc", "BuiltinObject"]

from dataclasses import dataclass, field
from collections import ChainMap
from functools import cached_property
from typing import Any


@dataclass
class BuiltinFunc:
    """A function the language provides, named by its FFSL type names."""

    name: str
    ptypes: list[str]
    rtype: str

    @cached_property
    def type(self) -> str:
        """The signature rendered as `(ptype, ...) -> rtype`."""
        ptypes = ", ".join(self.ptypes)
        return f"({ptypes}) -> {self.rtype}"


@dataclass
class BuiltinObject:
    """A name bound by the language rather than by the model.

    `type` is an FFSL type name, or one of the pseudo types
    `tile`, `src_tile` and `dst_tile` that a fire-model clause binds.
    `attrs` is filled in after parsing, once the tile variables are known.
    """

    name: str
    type: str
    value: Any | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


def add_builtins(scope: ChainMap[str, Any]) -> None:
    """Bind every builtin into `scope`, in place."""
    scope["exp"] = BuiltinFunc(name="exp", ptypes=["float"], rtype="float")

    scope["alignment"] = BuiltinFunc(
        name="alignment", ptypes=["position", "position", "float"], rtype="float"
    )
    scope["distance"] = BuiltinFunc(
        name="distance", ptypes=["position", "position"], rtype="float"
    )
