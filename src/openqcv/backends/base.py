"""The contract every backend follows."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.core.registry import Registry


class Backend(ABC):
    """Base class for anything that can run a :class:`QasmProgram`.

    To add one: subclass this, set ``name`` (and ``max_qubits`` if the
    device is limited), implement ``run`` and decorate the class with
    ``@register_backend``.

    Rules:
        * Read the circuit from ``program.qasm2`` or ``program.qasm3`` only,
          never from representation objects.
        * Return bitstrings in qubit order, as defined in
          :mod:`openqcv.core.program`.
    """

    name: ClassVar[str]
    max_qubits: ClassVar[int | None] = None

    @abstractmethod
    def run(
        self, program: QasmProgram, shots: int | None = None
    ) -> ExecutionResult:
        """Execute ``program``.

        Args:
            program: The program to run.
            shots: Number of samples; ``None`` means exact or the backend's
                default.
        """


BACKENDS: Registry[Backend] = Registry("backend", Backend)
register_backend = BACKENDS.register


class Provider(ABC):
    """A source of backends: a simulator package, a vendor, a device family.

    A backend is chosen as ``"provider:backend"``, e.g. ``"aer:statevector"``
    or ``"ibm_fake:fake_manila"``; a name without a provider is one of the
    built-in backends (provider ``"openqcv"``).

    To add one: subclass this, set ``name``, implement :meth:`backends` and
    :meth:`backend`, and decorate the class with ``@register_provider``. An
    installed package can instead declare it in its ``pyproject.toml``,
    and it is found without being imported::

        [project.entry-points."openqcv.providers"]
        mine = "my_package:MyProvider"

    Every backend a provider returns follows the :class:`Backend` rules: it
    reads only the program's QASM and returns qubit-order bitstrings.
    """

    name: ClassVar[str]

    @classmethod
    def available(cls) -> bool:
        """Return whether the packages this provider needs are installed."""
        return True

    @abstractmethod
    def backends(self) -> list[str]:
        """Return the names of the backends this provider offers."""

    @abstractmethod
    def backend(self, name: str, **options: Any) -> Backend:
        """Return the backend called ``name``, configured with ``options``.

        Raises:
            ValueError: If the provider has no backend of that name.
        """


PROVIDERS: Registry[Provider] = Registry(
    "provider", Provider, entry_point_group="openqcv.providers"
)
register_provider = PROVIDERS.register

PROVIDER_SEPARATOR = ":"
"""Separates the provider from the backend in a backend name."""


@register_provider
class OpenqcvProvider(Provider):
    """The backends built into openqcv: ``statevector`` and ``simulator``."""

    name = "openqcv"

    def backends(self) -> list[str]:
        """Return the registered backend names."""
        return BACKENDS.names()

    def backend(self, name: str, **options: Any) -> Backend:
        """Return a registered backend."""
        return BACKENDS.create(name, **options)


def get_backend(
    spec: str | type[Backend] | Backend, **options: Any
) -> Backend:
    """Return the backend named by ``spec``, configured with ``options``.

    Args:
        spec: ``"provider:backend"``; a built-in backend's name; or a
            :class:`Backend` class or instance.
        **options: Passed to the backend.

    Raises:
        ImportError: If the provider's packages are not installed.
        ValueError: If the provider or the backend is unknown.
    """
    if isinstance(spec, str) and PROVIDER_SEPARATOR in spec:
        provider, _, name = spec.partition(PROVIDER_SEPARATOR)
        return PROVIDERS.create(provider).backend(name, **options)
    return BACKENDS.create(spec, **options)


def list_providers() -> list[str]:
    """Return the providers whose packages are installed.

    A provider that an installed package declares but that fails to load is
    left out, with a warning that says why.
    """
    names = []
    for name in PROVIDERS.names():
        try:
            if PROVIDERS.get(name).available():
                names.append(name)
        except (ImportError, ValueError) as error:
            warnings.warn(
                f"Provider {name!r} could not be loaded: {error}",
                stacklevel=2,
            )
    return names
