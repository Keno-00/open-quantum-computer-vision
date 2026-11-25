"""Open Quantum: real QPUs and cloud simulators through one account.

`Open Quantum <https://openquantum.com>`_ (by Quantum Rings) runs circuits
on QPUs from IonQ, Rigetti, IQM and AQT, and on cloud simulators. Provider
``"openquantum"`` makes each of its backends an openqcv backend named
``"openquantum:<short code>"``, e.g. ``"openquantum:ionq:forte-1"``.

Installing
----------
::

    pip install "openqcv[openquantum]"

This installs Open Quantum's SDK with its Qiskit plugin
(``openquantum-sdk[qiskit]``), which openqcv uses to compile programs for
each device and to submit them.

Account and credentials
-----------------------
Create an account at https://openquantum.com, then an SDK key at
https://openquantum.com/keys ("Create SDK Key"). A key is a client ID
(``s_...``) and a client secret, which is shown only once. New accounts
receive free credits.

The secret spends your credits, so treat it like a password: it must not
end up in a file you commit, share, or publish (scripts, notebooks and
their outputs, logs, screenshots). The ways to give it to openqcv, safest
first:

1. **Save it once, in the system's credential store.** Install ``keyring``
   (``pip install keyring``), then save the key once from an interactive
   Python session, not from a script file::

       import getpass

       from openqcv.backends import openquantum

       openquantum.connect(
           client_id=input("client id: "),
           client_secret=getpass.getpass("client secret: "),
           save=True,
       )

   With ``keyring`` the secret goes to Windows Credential Manager, the
   macOS Keychain or the Secret Service on Linux, and every later session
   finds it with no code at all. Without ``keyring``, Open Quantum's SDK
   writes it in plain text to ``~/.openquantum/accounts.json``, which any
   program running as you can read.

2. **Environment variables**, set outside the code: in your shell profile,
   your IDE's run configuration, or your CI service's secret store. openqcv
   reads ``OPENQUANTUM_CLIENT_ID`` and ``OPENQUANTUM_CLIENT_SECRET``, or
   ``OPENQUANTUM_SDK_KEY`` (the path of the key's JSON file), or
   ``OPENQUANTUM_TOKEN`` (an access token). If you keep them in a ``.env``
   file, add it to ``.gitignore``.

3. **A key file outside the project**, e.g. ``~/.openquantum/sdk-key.json``
   with ``{"client_id": ..., "client_secret": ...}``:
   ``openquantum.connect(sdk_key_path=...)`` or ``OPENQUANTUM_SDK_KEY``.

4. **In your code**, as constants or variables, with :func:`connect`::

       import os

       from openqcv.backends import openquantum

       openquantum.connect(
           client_id=os.environ["MY_OQ_ID"],
           client_secret=os.environ["MY_OQ_SECRET"],
       )

   This works with literal strings too
   (``client_id="s_...", client_secret="..."``), but then the file holds
   the secret: keep it out of version control and never share it. Prefer
   reading the values from the environment or a secret manager, as above.

Without :func:`connect`, openqcv uses the saved account, then the
environment variables. If a secret is ever exposed, replace the key at
https://openquantum.com/keys. If you belong to several organizations, pass
``organization_id`` (below).

Choosing a backend
------------------
::

    import openqcv as qcv

    qcv.list_backends("openquantum")
    # ['openquantum:aqt:ibex-q1', 'openquantum:ibm:qiskit',
    #  'openquantum:ionq:forte-1', ..., 'openquantum:qr:sim', ...]

QPUs cost credits; the cloud simulators (``qr:sim``, ``ibm:qiskit``,
``xanadu:pennylane``, ``google:cirq``) are free while in beta, which makes
them the place to test a workflow first. :func:`backend_info` shows a
backend's type, status, queue and limits.

Running
-------
Any openqcv program runs as on the local backends::

    program = qcv.encode(image, "neqr")
    backend = qcv.backends.get_backend("openquantum:qr:sim", seed=7)
    result = qcv.run(program, backend, shots=1000)
    image_back = qcv.decode(result, program)

Create the backend once and pass the object, as above: creating it fetches
the device description. The program is compiled for the device (its native
gates, coupling map and qubit count) before it is sent; ``run`` waits until
the job finishes. QML models take the same backend
(``qml.VQC(..., backend=backend)``), but training sends one job per circuit
evaluation, so it is practical only on simulators or with small budgets.

Options
-------
``get_backend("openquantum:<code>", **options)`` accepts:

``seed``
    Seed of the compilation (qubit placement and routing), for repeatable
    circuits. Default ``None``.
``job_subcategory_id``
    The workload category Open Quantum records, as a short code. Default
    ``"oth:oth"`` (other); list them with
    ``OpenQuantumService().scheduler.get_job_categories()``.
``job_name``
    The jobs' name in the Open Quantum portal. Default ``"openqcv"``.
``organization_id``
    The organization billed. Default: the only one you belong to.
``configuration_data``
    Backend-specific settings, e.g. a simulator's engine:
    ``{"engine": "SCARLET_CPU", "precision": "DOUBLE"}``. The keys a
    backend takes are listed by
    ``OpenQuantumService().scheduler.configuration_options("<code>")``.
    ``shots`` is set by :func:`openqcv.run`.
``execution_plan``
    ``"auto"`` (default: the cheapest), ``"public"`` (shared, about half
    the price, the only plan free credits pay for) or ``"private"``.
``queue_priority``
    ``"auto"`` (default: the cheapest), ``"standard"``, ``"priority"`` or
    ``"instant"``; the last two cost extra.
``job_timeout_seconds``
    How long ``run`` waits for a job. Default 86400 (one day).
``verbose``
    Print each step of a submission (upload, quote, queue). Default
    ``False``.
``qasm_version``
    The OpenQASM version sent, 3 (default, as Open Quantum's own plugin
    sends) or 2.

``shots`` is passed to :func:`openqcv.run`; ``None`` means
:attr:`OpenQuantumBackend.default_shots`. A device's minimum and maximum are
checked before a job is sent.

Costs and limits
----------------
Every QPU job is quoted and the quote is approved automatically (the SDK's
default), so ``run`` spends credits; check the price of a backend in the
portal first. A failed or canceled job is refunded. An organization can
have at most 10 public and 100 private jobs active at once; a program may
be at most 10 MB of QASM.

Bit order and checking a backend
--------------------------------
Open Quantum returns counts as bit strings with classical bit 0 rightmost,
as Qiskit does; this is what Open Quantum's own Qiskit plugin reads, and
openqcv reverses them into qubit order. Open Quantum's documentation does
not state it, so :meth:`OpenQuantumBackend.check_bit_order` checks it with
one small job (free on a simulator), and openqcv's QIR decoding depends on
it.

Troubleshooting
---------------
``401``: the key or token is wrong or expired. ``402``: not enough
credits. ``409``: the organization's active-job limit is reached. A
``ValueError`` before submission names the device limit a program breaks
(qubits, shots).
"""

from __future__ import annotations

import os
from typing import Any

from qiskit import QuantumCircuit, transpile

from openqcv.backends.base import Backend, Provider, register_provider
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.export.qasm import load_circuit

INSTALL = 'pip install "openqcv[openquantum]"'

DEFAULT_SUBCATEGORY = "oth:oth"
"""Open Quantum's "other" workload category, its plugin's default."""

_SERVICE: list[Any] = []
"""The connected ``OpenQuantumService``, created on first use."""


def require_plugin() -> Any:
    """Return the ``openquantum_sdk_qiskit`` module.

    Raises:
        ImportError: If Open Quantum's SDK with its Qiskit plugin is not
            installed, with the install command.
    """
    try:
        import openquantum_sdk_qiskit
    except ImportError:
        raise ImportError(
            f"This provider needs Open Quantum's SDK: {INSTALL}"
        ) from None
    return openquantum_sdk_qiskit


def connect(
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    sdk_key_path: str | os.PathLike[str] | None = None,
    token: str | None = None,
    account_name: str | None = None,
    save: bool = False,
    **service_options: Any,
) -> Any:
    """Connect to Open Quantum for this process; return the service.

    Give at most one source of credentials; with none, the saved account
    is used, then the environment variables. See the module's "Account and
    credentials" for which source to prefer: a secret written in code ends
    up wherever the code goes.

    Args:
        client_id: The SDK key's client ID (``s_...``); needs
            ``client_secret``.
        client_secret: The SDK key's client secret.
        sdk_key_path: A JSON file with ``client_id`` and ``client_secret``.
        token: An access token, instead of a key.
        account_name: A saved account's name (the default one is
            ``"default"``).
        save: Also save the given key or token as ``account_name`` (default
            ``"default"``), in the system keyring when ``keyring`` is
            installed, so later sessions need no credentials in code.
        **service_options: Passed to ``OpenQuantumService``, e.g.
            ``accounts_file`` or service URLs.

    Raises:
        ImportError: If Open Quantum's SDK is not installed.
        ValueError: If only one of ``client_id`` and ``client_secret`` is
            given, several sources are given, or ``save`` has nothing to
            save. Messages never contain a secret.
    """
    plugin = require_plugin()
    if (client_id is None) != (client_secret is None):
        raise ValueError("client_id and client_secret must be given together")
    sources = [
        name
        for name, given in (
            ("client_id/client_secret", client_id is not None),
            ("sdk_key_path", sdk_key_path is not None),
            ("token", token is not None),
            ("account_name", account_name is not None and not save),
        )
        if given
    ]
    if len(sources) > 1:
        raise ValueError(
            f"Give one source of credentials, not {' and '.join(sources)}"
        )
    creds = None
    if client_id is not None:
        from openquantum_sdk.auth import ClientCredentials

        creds = ClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
    elif sdk_key_path is not None:
        from openquantum_sdk.credentials import load_creds_from_file

        creds = load_creds_from_file(os.fspath(sdk_key_path))
    if save:
        if creds is None and token is None:
            raise ValueError(
                "save=True needs client_id and client_secret, sdk_key_path "
                "or token"
            )
        plugin.OpenQuantumService.save_account(
            name=account_name or "default", creds=creds, token=token
        )
    if creds is not None or token is not None:
        service = plugin.OpenQuantumService(
            creds=creds, token=token, auto_load=False, **service_options
        )
    else:
        service = plugin.OpenQuantumService(
            account_name=account_name, **service_options
        )
    _SERVICE[:] = [service]
    return service


def service() -> Any:
    """Return the connected service, connecting with saved credentials."""
    if not _SERVICE:
        connect()
    return _SERVICE[0]


def backend_info(code: str) -> dict[str, Any]:
    """Return what Open Quantum reports about backend ``code``.

    Type (``"QPU"`` or ``"Simulator"``), status, whether it accepts jobs,
    its queue, and its limits (qubits per job, shots).
    """
    for entry in service().backends():
        if entry.get("short_code") == code:
            details = service().scheduler.get_backend_class(code)
            limits = (details.get("constraint_data") or {}).get("limits")
            return {**entry, "limits": limits or {}}
    raise ValueError(f"Unknown Open Quantum backend {code!r}")


class OpenQuantumBackend(Backend):
    """An Open Quantum backend running openqcv programs.

    The program's OpenQASM 2 is read, its registers merged into one (QPUs
    accept a single register; qubit ``j`` stays qubit ``j``), compiled for
    the device and submitted. Results are sampled counts in qubit order.
    See :mod:`openqcv.backends.openquantum` for the options.

    Args:
        device: Open Quantum's Qiskit backend for the device
            (``OpenQuantumService.return_backend``).
        name: The openqcv name, ``"openquantum:<code>"``.
        seed: Seed of the compilation.
        job_subcategory_id: Workload category.
        job_name: The jobs' name in the portal.
        organization_id: Organization billed.
        configuration_data: Backend-specific settings.
        execution_plan: ``"auto"``, ``"public"`` or ``"private"``.
        queue_priority: ``"auto"``, ``"standard"``, ``"priority"`` or
            ``"instant"``.
    """

    default_shots = 1000
    """Shots when :func:`openqcv.run` is given ``None``."""

    def __init__(
        self,
        device: Any,
        name: str,
        seed: int | None = None,
        job_subcategory_id: str = DEFAULT_SUBCATEGORY,
        job_name: str = "openqcv",
        organization_id: str | None = None,
        configuration_data: dict[str, Any] | None = None,
        execution_plan: str = "auto",
        queue_priority: str = "auto",
    ) -> None:
        self._device = device
        self.name = name
        self.seed = seed
        self.max_qubits = device.num_qubits or None
        self._submission = {
            "job_subcategory_id": job_subcategory_id,
            "name": job_name,
            "organization_id": organization_id,
            "configuration_data": dict(configuration_data or {}),
            "execution_plan": execution_plan,
            "queue_priority": queue_priority,
        }

    def run(
        self, program: QasmProgram, shots: int | None = None
    ) -> ExecutionResult:
        """Compile ``program`` for the device, run it and wait for counts.

        Raises:
            ValueError: If ``shots`` is less than 1, or the device's limits
                reject the program or the shots.
        """
        shots = self.default_shots if shots is None else shots
        if shots < 1:
            raise ValueError("shots must be at least 1")
        circuit = load_circuit(program)
        n = circuit.num_qubits
        single = QuantumCircuit(n, n)
        single.compose(circuit, qubits=range(n), clbits=range(n), inplace=True)
        compiled = transpile(
            single, backend=self._device, seed_transpiler=self.seed
        )
        job = self._device.run(compiled, shots=shots, **self._submission)
        counts = job.result().get_counts()
        # Classical bit j is rightmost (Qiskit's order) and measured qubit j.
        return ExecutionResult.from_counts(
            self.name,
            {
                bits.replace(" ", "").zfill(n)[::-1]: c
                for bits, c in counts.items()
            },
        )

    def check_bit_order(self, shots: int = 100) -> None:
        """Check with one job that results come back in qubit order.

        Runs X on qubit 0 of three qubits; the most frequent outcome must
        be ``"100"`` (qubit 0 set). On a QPU, noise leaves other outcomes,
        but far less frequent.

        Raises:
            AssertionError: If the outcome shows another bit order.
        """
        from openqcv.export.qasm import to_program
        from openqcv.representations.base import EncodedImage

        circuit = QuantumCircuit(3)
        circuit.x(0)
        program = to_program(EncodedImage(circuit, {}), "bit_order_check")
        counts = self.run(program, shots).counts
        top = max(counts, key=counts.get)
        if top != "100":
            raise AssertionError(
                f"{self.name} returned {top!r} most often for X on qubit 0; "
                "expected '100'. Its results are not in openqcv's bit order: "
                "do not decode them."
            )


@register_provider
class OpenQuantumProvider(Provider):
    """Open Quantum's QPUs and simulators; see the module for the guide.

    Raises:
        ImportError: If Open Quantum's SDK is not installed.
    """

    name = "openquantum"

    def __init__(self) -> None:
        require_plugin()

    @classmethod
    def available(cls) -> bool:
        """Return whether Open Quantum's SDK and its plugin are installed."""
        try:
            require_plugin()
        except ImportError:
            return False
        return True

    def backends(self) -> list[str]:
        """Return the short codes of Open Quantum's backends."""
        return sorted(
            entry["short_code"]
            for entry in service().backends()
            if entry.get("short_code")
        )

    def backend(
        self,
        name: str,
        seed: int | None = None,
        job_timeout_seconds: int | None = None,
        verbose: bool | None = None,
        qasm_version: int = 3,
        **options: Any,
    ) -> OpenQuantumBackend:
        """Return backend ``name`` (a short code such as ``"qr:sim"``).

        Raises:
            ValueError: If ``qasm_version`` is not 2 or 3, or Open Quantum
                has no such backend.
        """
        if qasm_version not in (2, 3):
            raise ValueError(
                f"qasm_version must be 2 or 3, got {qasm_version!r}"
            )
        device = service().return_backend(
            name,
            export_format=f"qasm{qasm_version}",
            verbose=verbose,
            job_timeout_seconds=job_timeout_seconds,
        )
        return OpenQuantumBackend(
            device, f"{self.name}:{name}", seed=seed, **options
        )
