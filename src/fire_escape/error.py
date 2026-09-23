"""Compiler exceptions, and the source position they carry."""

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
    """A location in an FFSL source file, counted from 1."""

    file: str
    line: int
    col: int


class Node(Protocol):
    """Anything carrying a source position, which is every AST node."""

    pos: Position


class CompilerError(RuntimeError):
    """Error in the compiler."""


class CodeError(RuntimeError):
    """Error in the code being compiled.

    A subclass sets `category`, which the exception argument opens.
    `str()` puts it in front of `message`, after the position when there is one.
    The `message` attribute holds the text without it.
    `pos` is filled in later by `node_error_attributer` when it is None here,
    so raising without a position is normal.
    """

    category: ClassVar[str] = "Code error"

    def __init__(self, message: str, pos: Position | None = None) -> None:
        super().__init__(f"{self.category}: {message}")

        self.message = message
        self.pos = pos

    def __str__(self) -> str:
        if self.pos is not None:
            return f"{self.pos.file}:{self.pos.line}:{self.pos.col}: {self.category}: {self.message}"
        else:
            return f"{self.category}: {self.message}"


def node_error_attributer[N: Node, **P, R](
    fn: Callable[Concatenate[N, P], R],
) -> Callable[Concatenate[N, P], R]:
    """Attribute an error raised inside `fn` to the node `fn` was given.

    A `CodeError` without a position gets the node's.
    Any exception that is neither a `CodeError` nor a `CompilerError`
    is rewrapped as a `CompilerError`.
    """

    @wraps(fn)
    def wrapper(node: N, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(node, *args, **kwargs)
        except CodeError as e:
            # An outer frame never overwrites the position an inner frame set.
            # The developer notes explain why the innermost position wins.
            if e.pos is None:
                e.pos = node.pos
            raise e
        except CompilerError:
            raise
        except Exception as e:
            # Anything else is a compiler fault,
            # as the developer notes on the two exception kinds explain.
            # The `from e` chain keeps its traceback.
            raise CompilerError(f"{node=}") from e

    return wrapper


class ParseError(CodeError):
    """A declaration that is missing, or declared more than once."""

    category = "Parse error"


# These two shadow the builtins of the same name.
# See the developer notes before catching either one.
class ReferenceError(CodeError):
    """A name that cannot be resolved, or is defined twice in one scope."""

    category = "Reference error"


class TypeError(CodeError):
    """Type error in the code being compiled."""

    category = "Type error"
