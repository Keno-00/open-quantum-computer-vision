"""The Open Quantum provider, against Open Quantum's own Qiskit plugin.

No account or network is used: the plugin's backend is built from explicit
device capabilities, as its documentation allows for tests, and its
scheduler is replaced by one that runs the submitted QASM on Qiskit's
reference simulator and returns counts as Open Quantum does (bit 0
rightmost, as its plugin reads them).
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister, qasm2, qasm3, transpile
from qiskit.providers.basic_provider import BasicSimulator

import openqcv as qcv
from openqcv.backends import PROVIDERS, get_backend, openquantum
from openqcv.export.qasm import to_program
from openqcv.representations import EncodedImage

plugin = pytest.importorskip(
    "openquantum_sdk_qiskit",
    reason="Open Quantum's SDK is not installed (openqcv[openquantum])",
)

SEED = 7
SHOTS = 2000
CODE = "test:qpu"
"""A made-up device: 5 qubits in a line, IBM-like native gates."""

CAPABILITIES = {
    "n_qubits": 5,
    "native_ops": [
        {"name": "rz", "arity": 1},
        {"name": "sx", "arity": 1},
        {"name": "x", "arity": 1},
        {"name": "cz", "arity": 2},
    ],
    "topology": {
        "directed_edges": False,
        "coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4]],
    },
    "limits": {"min_shots": 1, "max_shots": 10_000, "max_qubits_per_job": 5},
}


class FakeScheduler:
    """Runs submitted QASM locally; records each submission's settings."""

    def __init__(self, reverse=False):
        self.reverse = reverse
        self.submitted = []

    def submit_job(self, config, *, file_content=None, file_path=None):
        """Run the QASM; return a job holding its counts."""
        text = file_content.decode("utf-8")
        if text.startswith("OPENQASM 3"):
            circuit = qasm3.loads(text)
        else:
            circuit = qasm2.loads(
                text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
            )
        self.submitted.append((config, text, circuit))
        shots = config.configuration_data["shots"]
        counts = (
            BasicSimulator()
            .run(
                transpile(circuit, BasicSimulator()),
                shots=shots,
                seed_simulator=SEED,
            )
            .result()
            .get_counts()
        )
        if self.reverse:  # a platform that wrote bit 0 leftmost instead
            counts = {bits[::-1]: n for bits, n in counts.items()}
        return SimpleNamespace(counts=counts)

    def download_job_output(self, job):
        """Return the job's counts."""
        return job.counts

    def get_backend_class(self, code):
        """Return the device's capabilities."""
        return {"constraint_data": CAPABILITIES}


class FakeService:
    """The parts of OpenQuantumService the provider uses."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.calls = []

    def backends(self):
        """Return two backends, a QPU and a simulator."""
        return [
            {
                "short_code": CODE,
                "type": "QPU",
                "status": "Online",
                "accepting_jobs": True,
                "queue_depth": 0,
            },
            {
                "short_code": "test:sim",
                "type": "Simulator",
                "status": "Online",
                "accepting_jobs": True,
                "queue_depth": 0,
            },
        ]

    def return_backend(
        self, name, *, export_format, verbose, job_timeout_seconds
    ):
        """Return Open Quantum's backend for the made-up device."""
        self.calls.append((name, export_format))
        return plugin.OpenQuantumBackend(
            name=name,
            capabilities=CAPABILITIES,
            scheduler=self.scheduler,
            export_format=export_format,
        )


@pytest.fixture
def scheduler(monkeypatch):
    scheduler = FakeScheduler()
    monkeypatch.setattr(openquantum, "_SERVICE", [FakeService(scheduler)])
    return scheduler


def _program(qubits_set, qubits=3, registers=("r",)):
    sizes = [qubits - len(registers) + 1] + [1] * (len(registers) - 1)
    circuit = QuantumCircuit(
        *[
            QuantumRegister(size, name)
            for size, name in zip(sizes, registers, strict=True)
        ]
    )
    for qubit in qubits_set:
        circuit.x(qubit)
    return to_program(EncodedImage(circuit, {}), "test")


def test_lists_backends_and_their_details(scheduler):
    assert "openquantum" in qcv.list_providers()
    assert qcv.list_backends("openquantum") == [
        "openquantum:test:qpu",
        "openquantum:test:sim",
    ]
    info = openquantum.backend_info(CODE)
    assert info["type"] == "QPU"
    assert info["limits"]["max_qubits_per_job"] == 5


@pytest.mark.parametrize("version", [2, 3])
def test_qir_decodes_through_a_device(scheduler, version):
    pytest.importorskip("qiskit_qasm3_import") if version == 3 else None
    image = np.array([[3, 1], [2, 0]])
    program = qcv.encode(image, "neqr", bit_depth=2)
    backend = get_backend(
        f"openquantum:{CODE}", seed=SEED, qasm_version=version
    )
    assert backend.name == "openquantum:test:qpu"
    assert backend.max_qubits == 5
    result = qcv.run(program, backend, SHOTS)
    assert result.shots == SHOTS
    assert np.array_equal(qcv.decode(result, program), image)
    _, text, circuit = scheduler.submitted[-1]
    assert text.startswith(f"OPENQASM {version}")
    # Compiled for the device: its qubits, its native gates.
    assert circuit.num_qubits <= CAPABILITIES["n_qubits"]
    assert set(circuit.count_ops()) <= {
        "rz",
        "sx",
        "x",
        "cz",
        "measure",
        "barrier",
    }


def test_registers_are_merged_in_qubit_order(scheduler):
    program = _program([0, 2], qubits=4, registers=("a", "b", "c"))
    result = qcv.run(program, get_backend(f"openquantum:{CODE}"), SHOTS)
    assert result.counts == {"1010": SHOTS}


def test_submission_settings_reach_open_quantum(scheduler):
    backend = get_backend(
        f"openquantum:{CODE}",
        job_name="paper run",
        execution_plan="public",
        queue_priority="standard",
        configuration_data={"engine": "X"},
        job_subcategory_id="phys:oth",
        organization_id="org-1",
    )
    qcv.run(_program([0]), backend, 10)
    config, _, _ = scheduler.submitted[-1]
    assert config.name == "paper run"
    assert config.execution_plan == "public"
    assert config.queue_priority == "standard"
    assert config.job_subcategory_id == "phys:oth"
    assert config.organization_id == "org-1"
    assert config.configuration_data["engine"] == "X"
    assert config.configuration_data["shots"] == 10


def test_default_settings(scheduler):
    backend = get_backend(f"openquantum:{CODE}")
    result = qcv.run(_program([1]), backend)
    assert result.shots == openquantum.OpenQuantumBackend.default_shots
    config, text, _ = scheduler.submitted[-1]
    assert config.job_subcategory_id == openquantum.DEFAULT_SUBCATEGORY
    assert config.name == "openqcv"
    assert text.startswith("OPENQASM 3")


def test_check_bit_order_passes_and_catches_reversal(scheduler):
    backend = get_backend(f"openquantum:{CODE}")
    backend.check_bit_order()
    scheduler.reverse = True
    with pytest.raises(AssertionError, match="not in openqcv's bit order"):
        backend.check_bit_order()


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (
            lambda: get_backend(f"openquantum:{CODE}", qasm_version=4),
            ValueError,
            "qasm_version must be 2 or 3, got 4",
        ),
        (
            lambda: qcv.run(
                _program([]), get_backend(f"openquantum:{CODE}"), 0
            ),
            ValueError,
            "shots must be at least 1",
        ),
        (
            lambda: qcv.run(
                _program([]), get_backend(f"openquantum:{CODE}"), 20_000
            ),
            ValueError,
            "supports at most 10000 shots",
        ),
        (
            lambda: qcv.run(
                _program([], qubits=6), get_backend(f"openquantum:{CODE}")
            ),
            ValueError,
            "exceeds openquantum:test:qpu's limit of 5",
        ),
        (
            lambda: openquantum.backend_info("test:none"),
            ValueError,
            "Unknown Open Quantum backend 'test:none'",
        ),
    ],
)
def test_validation(scheduler, make, error, message):
    with pytest.raises(error, match=message):
        make()


def test_missing_sdk_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "openquantum_sdk_qiskit", None)
    assert "openquantum" not in qcv.list_providers()
    with pytest.raises(
        ImportError, match=r'pip install "openqcv\[openquantum\]"'
    ):
        PROVIDERS.create("openquantum")


# ------------------------------------------------------------ credentials

SECRET = "not-a-real-secret-7f3a"


class RecordingService:
    """Stands in for OpenQuantumService; records how it was created."""

    created = []
    saved = []

    def __init__(self, **kwargs):
        RecordingService.created.append(kwargs)

    @staticmethod
    def save_account(**kwargs):
        """Record a saved account instead of writing it."""
        RecordingService.saved.append(kwargs)


@pytest.fixture
def recording(monkeypatch):
    RecordingService.created, RecordingService.saved = [], []
    monkeypatch.setattr(plugin, "OpenQuantumService", RecordingService)
    monkeypatch.setattr(openquantum, "_SERVICE", [])
    return RecordingService


def test_connect_with_client_id_and_secret_from_code(recording):
    service = openquantum.connect(client_id="s_abc", client_secret=SECRET)
    (kwargs,) = recording.created
    assert kwargs["creds"].client_id == "s_abc"
    assert kwargs["creds"].client_secret == SECRET
    assert kwargs["auto_load"] is False  # code wins over saved accounts
    assert openquantum.service() is service
    assert recording.saved == []


def test_connect_with_a_key_file(recording, tmp_path):
    key = tmp_path / "sdk-key.json"
    key.write_text(
        json.dumps({"client_id": "s_file", "client_secret": SECRET})
    )
    openquantum.connect(sdk_key_path=key)
    assert recording.created[0]["creds"].client_id == "s_file"


def test_connect_with_a_token_or_a_saved_account(recording):
    openquantum.connect(token="jwt")
    assert recording.created[-1]["token"] == "jwt"
    openquantum.connect(account_name="lab")
    assert recording.created[-1] == {"account_name": "lab"}
    openquantum.connect()
    assert recording.created[-1] == {"account_name": None}


def test_connect_can_save_the_key(recording):
    openquantum.connect(
        client_id="s_abc",
        client_secret=SECRET,
        save=True,
        account_name="lab",
    )
    (saved,) = recording.saved
    assert saved["name"] == "lab"
    assert saved["creds"].client_secret == SECRET


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"client_id": "s_abc"}, "must be given together"),
        ({"client_secret": SECRET}, "must be given together"),
        (
            {"client_id": "s_abc", "client_secret": SECRET, "token": "jwt"},
            "Give one source of credentials, not client_id/client_secret "
            "and token",
        ),
        (
            {"token": "jwt", "account_name": "lab"},
            "not token and account_name",
        ),
        ({"save": True}, "save=True needs client_id and client_secret"),
    ],
)
def test_connect_validation_never_shows_the_secret(
    recording, options, message
):
    with pytest.raises(ValueError, match=message) as error:
        openquantum.connect(**options)
    assert SECRET not in str(error.value)
    assert recording.created == []
