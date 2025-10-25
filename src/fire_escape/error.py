"""Standardized Exceptions."""

__all__ = ["Position", "CompilerError", "CodeError", "ParseError", "ReferenceError", "TypeError"]

from typing import ClassVar
from dataclasses import dataclass


@dataclass
class Position:
    file: str
    line: int
    col: int


class CompilerError(RuntimeError):
    """Error in the ffsc compiler."""


class CodeError(RuntimeError):
    """Error in the code being compiled."""

    category: ClassVar[str] = "Code error"

    def __init__(self, message: str, pos: Position | None = None):
        super().__init__(f"{self.category}: {message}")

        self.message = message
        self.pos = pos

    def __str__(self):
        if self.pos is not None:
            return f"{self.pos.file}:{self.pos.line}:{self.pos.col}: {self.category}: {self.message}"
        else:
            return f"{self.category}: {self.message}"


class ParseError(CodeError):
    """Parse error."""

    category = "Parse error"


class ReferenceError(CodeError):
    """Reference error."""

    category = "Reference error"


class TypeError(CodeError):
    """Reference error."""

    category = "Type error"
