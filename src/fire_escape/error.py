"""Standardized Exceptions."""

__all__ = [
    "Position",
    "CompilerError",
    "CodeError",
    "ParseError",
    "ReferenceError",
    "TypeError",
    "node_error_attributer",
]

from typing import Protocol, Callable, Concatenate, ClassVar
from dataclasses import dataclass
from functools import wraps


@dataclass
class Position:
    file: str
    line: int
    col: int


class Node(Protocol):
    pos: Position


class CompilerError(RuntimeError):
    """Error in the compiler."""


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


def node_error_attributer[N: Node, **P, R](
    fn: Callable[Concatenate[N, P], R],
) -> Callable[Concatenate[N, P], R]:
    @wraps(fn)
    def wrapper(node: N, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(node, *args, **kwargs)
        except CodeError as e:
            if e.pos is None:
                e.pos = node.pos
            raise e
        except CompilerError:
            raise
        except Exception as e:
            raise CompilerError(f"{node=}") from e

    return wrapper


class ParseError(CodeError):
    """Parse error."""

    category = "Parse error"


class ReferenceError(CodeError):
    """Reference error."""

    category = "Reference error"


class TypeError(CodeError):
    """Reference error."""

    category = "Type error"
