"""Standardized Exceptions."""

__all__ = ["CompilerError", "CodeError", "ParseError", "ReferenceError", "TypeError"]

from pathlib import Path
from typing import ClassVar


class CompilerError(RuntimeError):
    """Error in the ffsc compiler."""


class CodeError(RuntimeError):
    """Error in the code being compiled."""

    category: ClassVar[str] = "Code error"

    def __init__(
        self,
        message: str,
        line: int | None = None,
        col: int | None = None,
        file: Path | str | None = None,
    ):
        super().__init__(f"{self.category}: {message}")

        self.message = message
        self.line = line
        self.col = col
        self.file = file

    def __str__(self):
        if self.file is not None and self.line is not None and self.col is not None:
            return (
                f"{self.file}:{self.line}:{self.col}: {self.category}: {self.message}"
            )
        elif self.line is not None and self.col is not None:
            return f"{self.line}:{self.col}: {self.category}: {self.message}"
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
