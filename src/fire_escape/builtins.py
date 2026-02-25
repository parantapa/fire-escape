"""Builtin functions and constants."""

from __future__ import annotations

__all__ = ["BuiltinFunc", "BuiltinObject"]

from dataclasses import dataclass, field
from collections import ChainMap
from functools import cached_property
from typing import Any


@dataclass
class BuiltinFunc:
    name: str
    ptypes: list[str]
    rtype: str

    @cached_property
    def type(self) -> str:
        ptypes = ", ".join(self.ptypes)
        return f"({ptypes}) -> {self.rtype}"


@dataclass
class BuiltinObject:
    name: str
    type: str
    value: Any | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


def add_builtins(scope: ChainMap[str, Any]):
    scope["exp"] = BuiltinFunc(name="exp", ptypes=["float"], rtype="float")

    scope["alignment"] = BuiltinFunc(name="alignment", ptypes=["position", "position", "float"], rtype="float")
    scope["distance"] = BuiltinFunc(name="distance", ptypes=["position", "position"], rtype="float")
