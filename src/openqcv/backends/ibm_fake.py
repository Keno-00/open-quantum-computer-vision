"""IBM's simulated devices: device noise, gates and coupling, run on Aer.

Needs ``qiskit-ibm-runtime`` and ``qiskit-aer`` (``pip install
"openqcv[ibm]"``). Each device of ``qiskit_ibm_runtime.fake_provider`` is a
backend, e.g. ``"ibm_fake:fake_manila"``: programs are transpiled to the
device and sampled with its noise model, so results show what the device's
errors do to a QIR. A device's qubit count limits the programs it runs.
"""

from __future__ import annotations

import importlib
from typing import Any

from openqcv.backends.aer import AerBackend, require_aer
from openqcv.backends.base import Provider, register_provider

INSTALL = 'pip install "openqcv[ibm]"'


_DEVICES: dict[str, Any] = {}
"""Device name -> IBM's fake device, loaded on first use: building all of
them reads every device's configuration."""


def _devices() -> dict[str, Any]:
    if not _DEVICES:
        try:
            from qiskit_ibm_runtime.fake_provider import (
                FakeProviderForBackendV2,
            )
        except ImportError:
            raise ImportError(
                f"This provider needs qiskit-ibm-runtime: {INSTALL}"
            ) from None
        devices = FakeProviderForBackendV2().backends()
        _DEVICES.update({device.name: device for device in devices})
    return _DEVICES


@register_provider
class IBMFakeProvider(Provider):
    """IBM's simulated devices (``fake_manila``, ``fake_sherbrooke``, ...).

    ``backend(name, seed=..., **options)`` returns an Aer simulator of the
    device; ``options`` go to ``AerSimulator.from_backend``.

    Raises:
        ImportError: If qiskit-ibm-runtime or qiskit-aer is not installed.
    """

    name = "ibm_fake"

    def __init__(self) -> None:
        require_aer()
        self._devices = _devices()

    @classmethod
    def available(cls) -> bool:
        """Return whether qiskit-ibm-runtime and qiskit-aer are installed."""
        try:
            require_aer()
            importlib.import_module("qiskit_ibm_runtime")
        except ImportError:
            return False
        return True

    def backends(self) -> list[str]:
        """Return the simulated devices' names."""
        return sorted(self._devices)

    def backend(
        self, name: str, seed: int | None = None, **options: Any
    ) -> AerBackend:
        """Return an Aer simulator of device ``name``.

        Raises:
            ValueError: If there is no such device.
        """
        if name not in self._devices:
            raise ValueError(
                f"Unknown device {name!r}; list them with "
                "list_backends('ibm_fake')"
            )
        backend = AerBackend(seed=seed, target=self._devices[name], **options)
        backend.name = f"{self.name}:{name}"
        return backend
