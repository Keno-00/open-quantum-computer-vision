"""Backends that run a :class:`~openqcv.core.program.QasmProgram`.

Backends come from providers: ``"openqcv"`` (built in), ``"aer"`` (Qiskit
Aer), ``"ibm_fake"`` (IBM's simulated devices), ``"openquantum"`` (Open
Quantum's QPUs and cloud simulators), and any installed package that
declares one (see :class:`~openqcv.backends.base.Provider`).
"""

from openqcv.backends import aer, ibm_fake, openquantum
from openqcv.backends.base import (
    BACKENDS,
    PROVIDERS,
    Backend,
    Provider,
    get_backend,
    list_providers,
    register_backend,
    register_provider,
)
from openqcv.backends.simulator import SimulatorBackend
from openqcv.backends.statevector import StatevectorBackend

__all__ = [
    "BACKENDS",
    "PROVIDERS",
    "Backend",
    "Provider",
    "SimulatorBackend",
    "StatevectorBackend",
    "aer",
    "get_backend",
    "ibm_fake",
    "list_providers",
    "openquantum",
    "register_backend",
    "register_provider",
]
