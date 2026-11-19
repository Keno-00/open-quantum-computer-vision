# Design

How openqcv is built and the rules that keep it that way. The project goals
these rules serve are in the [README](../README.md#goals).

## Repository map

```
src/openqcv/
├── core/              QasmProgram, ExecutionResult, Block, Registry
├── image/             as_grayscale, position_qubits, load_image
├── representations/   Representation base, QPIE, FRQI, NEQR
├── export/            qasm.py (lowering), logical.py (textbook OpenQASM 3)
├── backends/          Backend base, statevector, simulator
├── viz/               circuit.py, image.py, drawing.py (PAPER_STYLE)
└── pipeline.py        encode, run, decode
tests/                 test_roundtrip.py, test_logical.py, test_core.py
docs/                  design.md, make_figures.py, images/
```

## Architecture invariants

Changing any of these is a design change and needs the maintainer's
approval.

1. Dependency direction: `core` imports nothing from openqcv.
   `representations` and `backends` never import each other; `pipeline` is
   the only module that knows both. `viz` reads QASM text, not
   representation objects.
2. `QasmProgram` is the contract between representations and backends.
   Everything needed to run, draw or decode an image is in it, and it
   survives `to_json` / `from_json` unchanged.
3. `decode` is a classmethod and may use only the result and the program,
   never encoding options, so a received program is decodable by anyone.
4. Backends read only `program.qasm2` / `program.qasm3` and return
   bitstrings in the order defined in `openqcv.core.program`.
5. Register names are roles (`position`, `color`, `intensity`).
6. Representations are built from standard gates, using `controlled` for
   conditioned gates and a `Block` with a unique name for each high-level
   step. Opaque library gates (`StatePreparation`, `UCRYGate`, ...) are not
   allowed: logical QASM cannot express them, so the figures would omit
   what the circuit does.
7. `export.logical.dumps` and `export.logical.parse` agree on the subset
   defined in that module.
8. Lowering never assumes clean ancillas (`qubits_initially_zero=False`).
   Blocks can sit anywhere in a circuit, and borrowing a qubit that is in
   use corrupts the state without any error.

## Code

- PEP 8 and PEP 257, checked by ruff as configured in `pyproject.toml`.
  Rules are not silenced with `noqa`; the code is fixed, or the
  configuration is changed with its reason written next to it.
- If the formatter produces an awkward wrap, restructure the code (name an
  intermediate value, move a comment onto its own line).
- Public names are listed in `__all__`; private helpers start with `_`.
- Inputs are validated where they enter (public functions, constructors).
  Bad values raise `ValueError`, bad types `TypeError`, with a message that
  says what was expected. A silently wrong result is never acceptable.
- A rule that can be expressed in code (types, classmethods, registries,
  validation) is expressed in code, not in a comment.
- No duplicated logic and no unnamed constants.
- Measure rather than estimate. A tolerance gets a name and a reason.
- Public functions are fully annotated (`from __future__ import
  annotations`, builtin generics, `X | None`, `numpy.typing.ArrayLike` for
  images).
- Python 3.10+ only syntax, and no deprecated dependency APIs: the test
  suite turns `DeprecationWarning` into errors.
- Dependencies are numpy, qiskit and matplotlib (Pillow optional). Adding
  one needs the maintainer's approval.

## Documentation

Every piece of information has exactly one home: the place closest to where
it is needed. Other places link to it instead of restating it, so nothing can
drift out of date or carry a concept into a document it does not belong to.

| Information | Home |
|---|---|
| What openqcv is, how to install and start | `README.md` |
| How to contribute: setup, checks, pull requests | `CONTRIBUTING.md` |
| Architecture, invariants, these rules | `docs/design.md` |
| How AI agents must work in this repository | `AGENTS.md` |
| What a module, class or function does, its conventions | its docstring |
| The paper or standard behind an implementation | `References:` in the docstring of the code that implements it |

Weight is earned, not assigned:

- A sentence stays only if a reader needs it to use, change or trust the
  code. Promotional wording, restated headings and summaries of what was
  just said are removed.
- Emphasis (bold, "must", "never") is used only where ignoring the point
  causes a defect, and the text says what that defect is.
- Each work is cited once, where it is used, and checked against its
  publisher record (DOI or arXiv identifier) before it is added.

## Figures

- Everything drawn uses `PAPER_STYLE` (black and white, Computer Modern via
  mathtext, inward ticks).
- Figures carry information: real gate symbols and angles, pixel values,
  error bars, test statistics.
- A changed figure is rendered and looked at before the change is done.

## Tests

- Every representation round-trips exactly on `statevector` (including a
  padded, asymmetric image, which exposes bit-order errors), through JSON,
  and within a stated tolerance on `simulator`.
- Every bug fix comes with a test that fails without it.
- Every validation error has a test asserting its type and message.
- Assertions are exact where possible; random tests fix their seed.

## Adding a representation

1. Create `representations/<name>.py`: a `Representation` subclass
   decorated with `@register_representation`, whose docstring gives the
   state, qubit count, circuit, readout and `References:`.
2. Build the circuit with `controlled` and named `Block`s.
3. Put everything decoding needs in `meta`; implement `decode` as a
   classmethod using `marginal` and `image_shape`.
4. Export the class in `representations/__init__.py`.
5. Add it to `REPRESENTATIONS` in `tests/test_roundtrip.py` and add its
   validation tests to `tests/test_core.py`.
6. Render its circuit and look at it.

## Adding a backend

1. Create `backends/<name>.py`: a `Backend` subclass decorated with
   `@register_backend`, with `max_qubits` set if the device is limited.
2. Read only the QASM text; convert results to qubit order; build sampled
   results with `ExecutionResult.from_counts`.
3. Validate `shots` explicitly: `None` means exact or default, and `0` is
   an error.
4. Add round-trip tests, skipped with a stated reason when credentials or
   hardware are unavailable.
