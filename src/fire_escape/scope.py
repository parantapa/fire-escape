"""Program scope."""

from __future__ import annotations

from typing import Any, overload
from dataclasses import dataclass, field


@dataclass
class Scope:
    """Progrma scope."""

    name: str
    names: dict[str, Any] = field(default_factory=dict)
    parent: Scope | None = field(default=None)

    def __rich_repr__(self):
        yield "name", self.name
        yield "names", {k: type(v).__name__ for k, v in self.names.items()}
        if self.parent is not None:
            yield "parent", self.parent.name

    @overload
    def resolve[T](self, k: str, type: type[T]) -> T: ...

    @overload
    def resolve[T](self, k: str, type: tuple[type[T], ...]) -> T: ...

    @overload
    def resolve(self, k: str, type: Any | None) -> Any: ...

    @overload
    def resolve(self, k: str) -> Any: ...

    def resolve(self, k, type=None):
        if k in self.names:
            ret = self.names[k]
        elif self.parent is not None:
            ret = self.parent.resolve(k, type)  # type: ignore
        else:
            raise RuntimeError(f"{k} is not defined.")

        if type is None:
            return ret

        if not isinstance(ret, type):
            raise RuntimeError(f"Expected object of type {type}; got {type(ret)}")

        return ret

    def define(self, k: str, v: Any):
        if k in self.names:
            raise RuntimeError(f"{k} has been already defined.")
        self.names[k] = v

    def undef(self, k: str):
        del self.names[k]

    def is_defined(self, k: str) -> bool:
        if k in self.names:
            return True
        elif self.parent is not None:
            return self.parent.is_defined(k)
        else:
            return False

    def root(self) -> Scope:
        ret = self
        while ret.parent is not None:
            ret = ret.parent
        return ret
