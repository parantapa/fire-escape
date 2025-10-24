"""Builtin functions and constants."""

from __future__ import annotations

__all__ = ["BuiltinFunc", "BuiltinConst"]

from dataclasses import dataclass
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
class BuiltinConst:
    name: str
    type: str


def add_builtins(scope: ChainMap[str, Any]):
    scope["sqrt"] = BuiltinFunc(name="sqrt", ptypes=["float"], rtype="float")

    scope["CONFIG"] = BuiltinConst(name="CONFIG", type="json")
