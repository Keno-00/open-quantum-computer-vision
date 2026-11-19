# AGENTS.md

Instructions for anyone changing openqcv: AI coding agents and humans alike.
Read the [Principles in README.md](README.md#principles) first; this file
turns them into concrete rules. If a rule here blocks something you think is
right, stop and ask the maintainer rather than working around it.

## Working agreement

- **The maintainer drives the design.** Implement what was asked. Do not add
  features, dependencies, configuration, CI or files that were not requested;
  propose them instead.
- **Ask before anything hard to undo or outward-facing:** rewriting git
  history, force-pushing, publishing, deleting files you did not create.
- **Report honestly.** If tests fail or something was not verified, say so
  with the evidence. Never weaken a test to make it pass.

## Repository map

```
src/openqcv/
├── core/              QasmProgram, ExecutionResult (the shared data), Block, Registry
├── image/             as_grayscale, position_qubits, load_image (classical side)
├── representations/   Representation base + QPIE, FRQI, NEQR; helpers controlled(), marginal()
├── export/            qasm.py: EncodedImage -> QasmProgram; logical.py: textbook OpenQASM 3 read/write
├── backends/          Backend base + statevector (exact), simulator (shots)
├── viz/               circuit.py (quantikz-style), image.py, drawing.py (PAPER_STYLE)
└── pipeline.py        encode / run / decode: the public workflow
tests/                 test_roundtrip.py, test_logical.py, test_core.py
docs/make_figures.py   regenerates README figures
```

## Architecture invariants

These are load-bearing. Breaking one is a design change and needs the
maintainer's approval.

1. **Dependency direction.** `core` imports nothing from openqcv.
   `representations` and `backends` import `core` (and `image`/`export`
   helpers) but **never each other**. `pipeline` is the only module that
   knows both sides. `viz` reads QASM text, not representation objects.
2. **`QasmProgram` is the contract.** Everything needed to run, draw or decode
   an image is in it, and it survives `to_json` / `from_json` unchanged.
   `meta` must be JSON-serialisable.
3. **`decode` is a classmethod** and may use only the `ExecutionResult` and
   `program.meta` / `program.registers`, never encoding options. This is
   what makes a received program decodable by anyone.
4. **Backends read QASM text only** (`program.qasm2` / `program.qasm3`) and
   return bitstrings in **qubit order** (character `j` = qubit `j`).
5. **Register names are roles** (`position`, `color`, `intensity`); within a
   register the first qubit is the least significant bit.
6. **Representations are built from standard gates.** Use
   `controlled(circuit, gate, controls, value, target)` for conditioned gates
   (open controls come from 0-bits of `value`) and wrap each high-level step
   in a `Block` with a unique, meaningful name (`pixel_5`, `level_0`).
   Never use opaque library gates (`StatePreparation`, `UCRYGate`, ...):
   logical QASM cannot express them and the figures would lie by omission.
7. **Logical QASM is a defined subset of OpenQASM 3** (see the
   `openqcv.export.logical` docstring). `dumps` and `parse` must agree:
   anything one writes, the other reads.
8. **Lowering never assumes clean ancillas** (`qubits_initially_zero=False`):
   blocks can sit anywhere in a circuit, and borrowing a "free" qubit that is
   actually in use corrupts the state silently.

## Code design rules

- **PEP 8 and PEP 257, enforced.** Lines are at most 79 characters.
  `ruff format` and `ruff check` (configured in `pyproject.toml`: PEP 8,
  naming, import order, Google-style docstrings, pyflakes, bugbear,
  pyupgrade) must both pass. Do not silence a rule with `noqa`; fix the
  code, or change the configuration with a stated reason.
- **Readable after formatting.** If the formatter produces an awkward wrap,
  restructure the code (name an intermediate value, move the comment onto
  its own line) instead of accepting it.
- **Small, explicit public API.** Public names are listed in `__all__`.
  Private helpers start with `_`. Prefer a function over a class unless state
  or polymorphism is needed.
- **Validate at the boundary, fail loudly and specifically.** Check inputs
  where they enter (public functions, constructors) and raise `ValueError`
  for bad values or `TypeError` for bad types, with a message that says what
  was expected and how to fix it. Never return a silently wrong result.
- **Enforce rules in code, not comments.** If a rule can be expressed with
  types, classmethods, registries or validation, do that.
- **One source of truth.** No duplicated logic (shared decoding goes through
  `marginal`, shapes through `image_shape`) and no magic numbers: give
  constants names and units.
- **Exact over heuristic.** Measure instead of estimating (e.g. text widths
  in `viz.circuit` come from `TextPath`). If a tolerance is needed, name it
  and say why.
- **Typing.** Every public function is fully annotated, using
  `from __future__ import annotations`, builtin generics (`list[int]`),
  `X | None`, and `numpy.typing.ArrayLike` for image inputs. The package
  ships `py.typed`.
- **Docstrings** use Google style (`Args:`, `Returns:`, `Raises:`). Module
  docstrings explain *why* the module exists; comments explain *why*, not
  *what*.
- **Compatibility.** Code must run on Python 3.10+ (no syntax newer than
  that, e.g. no backslashes inside f-string expressions) and must not use
  deprecated dependency APIs: the test suite turns `DeprecationWarning` into
  errors.
- **Dependencies** are numpy, qiskit and matplotlib (Pillow optional). Adding
  one requires the maintainer's approval.

## Figures

- Use `PAPER_STYLE` (`rc_context`) for everything drawn: black and white,
  Computer Modern via mathtext, inward ticks.
- Figures must carry information, not decoration: real gate symbols and
  angles, pixel values, error bars, test statistics.
- Look at every figure you change (render it to PNG and inspect it) before
  calling the work done.

## Testing

- Before finishing any change, run from the repository root:
  `ruff format --check .`, `ruff check .` and `pytest`. All must pass.
- **Every representation** must round-trip exactly on the `statevector`
  backend (including a padded, asymmetric image that exposes bit-order
  errors), through JSON, and within stated tolerance on the `simulator`.
- **Every bug fix** gets a test that fails without the fix.
- **Every validation error** has a test asserting its type and message.
- Prefer exact assertions; when randomness is involved, fix the seed.

## Adding a representation

1. Create `representations/<name>.py` with a class decorated by
   `@register_representation`, a docstring giving the state, qubit count,
   circuit and readout, and a reference to the paper.
2. Build the circuit from standard gates with `controlled` and named `Block`s.
3. Put everything decoding needs in `meta`; make `decode` a classmethod using
   `marginal` and `image_shape`.
4. Export it in `representations/__init__.py`.
5. Add it to `REPRESENTATIONS` in `tests/test_roundtrip.py` and add its
   validation tests to `tests/test_core.py`.
6. Add a row to the README table, regenerate figures, and look at its circuit.

## Adding a backend

1. Create `backends/<name>.py` with a `Backend` subclass decorated by
   `@register_backend`; set `max_qubits` if the device is limited.
2. Read only `program.qasm2` or `program.qasm3`; convert results to qubit
   order; build results with `ExecutionResult.from_counts` when sampling.
3. Validate `shots` explicitly (`None` means default or exact; never treat
   `0` as "default").
4. Add round-trip tests (skipped with a clear reason if credentials or
   hardware are unavailable) and a README row.
