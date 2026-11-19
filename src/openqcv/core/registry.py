"""A name -> class registry, used for representations and backends."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps short names such as ``"qpie"`` to subclasses of ``base``.

    This is what lets users pick a representation or backend with a plain
    string, while still accepting a class or a configured instance.

    Args:
        kind: Human-readable name used in error messages.
        base: Every registered class must subclass it.
    """

    def __init__(self, kind: str, base: type[T]) -> None:
        self.kind = kind
        self.base = base
        self._classes: dict[str, type[T]] = {}

    def register(self, cls: type[T]) -> type[T]:
        """Class decorator: register ``cls`` under its ``name`` attribute."""
        if not (isinstance(cls, type) and issubclass(cls, self.base)):
            raise TypeError(
                f"Only subclasses of {self.base.__name__} can be registered"
            )
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{cls.__name__} needs a non-empty class attribute 'name'"
            )
        key = name.lower()
        if key in self._classes:
            raise ValueError(
                f"A {self.kind} called {name!r} is already registered"
            )
        self._classes[key] = cls
        return cls

    def names(self) -> list[str]:
        """All registered names, sorted."""
        return sorted(self._classes)

    def get(self, name: str) -> type[T]:
        """The class registered under ``name`` (case-insensitive)."""
        try:
            return self._classes[name.lower()]
        except KeyError:
            available = ", ".join(self.names())
            raise ValueError(
                f"Unknown {self.kind} {name!r}. Available: {available}"
            ) from None

    def create(self, spec: str | type[T] | T, **options: Any) -> T:
        """An instance from a name or class plus options, or ``spec`` itself.

        Raises:
            TypeError: If ``spec`` is an instance and options are given, or
                ``spec`` is not a name, class or instance of ``base``.
        """
        if isinstance(spec, str):
            return self.get(spec)(**options)
        if isinstance(spec, type) and issubclass(spec, self.base):
            return spec(**options)
        if isinstance(spec, self.base):
            if options:
                raise TypeError(
                    f"Options can't be applied to an existing {self.kind} "
                    "instance; configure it when you create it"
                )
            return spec
        raise TypeError(
            f"Expected a {self.kind} name, class or instance, "
            f"got {type(spec).__name__}"
        )
