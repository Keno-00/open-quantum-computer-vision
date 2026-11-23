"""A registry of classes by name, for representations and backends."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A mapping from short names such as ``"qpie"`` to subclasses of ``base``.

    This is what lets users pick a representation or backend with a plain
    string, while still accepting a class or a configured instance.

    Installed packages can add classes without being imported: a class
    named by an entry point in ``entry_point_group`` is loaded the first
    time its name is looked up. A package declares one in its
    ``pyproject.toml``::

        [project.entry-points."openqcv.representations"]
        mine = "my_package:Mine"

    Args:
        kind: Human-readable name used in error messages.
        base: Every registered class must subclass it.
        entry_point_group: The entry point group searched for names that
            are not registered, or ``None`` to search none.
    """

    def __init__(
        self,
        kind: str,
        base: type[T],
        entry_point_group: str | None = None,
    ) -> None:
        self.kind = kind
        self.base = base
        self.entry_point_group = entry_point_group
        self._classes: dict[str, type[T]] = {}

    def register(self, cls: type[T]) -> type[T]:
        """Register ``cls`` under its ``name`` attribute; a class decorator."""
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
        """Return all registered and installed names, sorted."""
        installed = {entry.name.lower() for entry in self._entry_points()}
        return sorted(installed | self._classes.keys())

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``, ignoring case.

        Raises:
            ValueError: If no class is registered or installed under
                ``name``, or an entry point loads a class with another name.
        """
        key = name.lower()
        if key not in self._classes:
            self._load(key)
        try:
            return self._classes[key]
        except KeyError:
            available = ", ".join(self.names())
            raise ValueError(
                f"Unknown {self.kind} {name!r}. Available: {available}"
            ) from None

    def create(self, spec: str | type[T] | T, **options: Any) -> T:
        """Return an instance for ``spec``, created with ``options`` if needed.

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

    def _entry_points(self) -> list[Any]:
        if self.entry_point_group is None:
            return []
        return list(entry_points(group=self.entry_point_group))

    def _load(self, key: str) -> None:
        """Import the entry point named ``key`` and register its class."""
        for entry in self._entry_points():
            if entry.name.lower() != key:
                continue
            cls = entry.load()
            # Importing the module may already have registered the class.
            if cls not in self._classes.values():
                self.register(cls)
            if self._classes.get(key) is not cls:
                raise ValueError(
                    f"Entry point {entry.name!r} ({entry.value}) loads a "
                    f"{self.kind} named {cls.name!r}; the names must match"
                )
            return
