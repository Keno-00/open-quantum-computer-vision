"""Quantikz-style circuit drawings from logical OpenQASM 3.

Black and white, textbook notation: filled and open controls, ⊕ targets,
``R_y(θ)`` boxes with their angles, meters, classical double wires and
classically controlled gates, and dashed frames around each named gate
definition (e.g. one per pixel).

Drawing works from the QASM text only, so any program, including one loaded
from JSON or a hand-written logical QASM string, can be drawn.

Coordinates are in *lane units*: lane ``k`` is the horizontal line
``y = -k``, and one unit is :data:`INCH_PER_UNIT` inches.

References:
    A. Kay, "Tutorial on the Quantikz Package" (2018). arXiv:1809.03842
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from openqcv.core.program import QasmProgram
from openqcv.export.logical import LogicalCircuit, LogicalOp, parse
from openqcv.viz.drawing import PAPER_STYLE, Drawing

INCH_PER_UNIT = 0.42
FONT_SIZE = 11  # gate labels, kets and register names (points)
SMALL_FONT = 8  # qubit indices and group labels
BOX = 0.56  # gate box height and minimum width
BOX_PADDING = 0.3  # horizontal space around a gate label
GAP = 0.34  # space between columns
GROUP_PAD = 0.14  # dashed frame margin around its gates
WIRE = 0.9  # quantum wire and gate outline width (points)
CLASSICAL = 0.7  # classical double-wire line width (points)
DOT = 0.075  # control dot radius
TARGET = 0.17  # ⊕ radius
DOUBLE = 0.035  # half the distance between classical double lines
LABEL_GAP = 0.2  # space between left-hand labels (ket, index, name)
DEFAULT_INPUT = r"|0\rangle"

_SYMBOLS = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": r"S^\dagger",
    "t": "T",
    "tdg": r"T^\dagger",
    "sx": r"\sqrt{X}",
    "id": "I",
    "p": "P",
    "rx": "R_x",
    "ry": "R_y",
    "rz": "R_z",
}


def draw(
    program: QasmProgram | str, inputs: Mapping[str, str] | None = None
) -> Drawing:
    r"""Draw a program's logical circuit, or a logical OpenQASM 3 string.

    Args:
        program: A :class:`QasmProgram` or logical OpenQASM 3 text.
        inputs: Optional input-state label per register, in mathtext, e.g.
            ``{"alice": r"|\psi\rangle"}``. The default is ``|0⟩``.
    """
    text = program if isinstance(program, str) else program.logical_qasm
    with rc_context(PAPER_STYLE):
        return Drawing(_render(parse(text), dict(inputs or {})))


# ------------------------------------------------------------------- layout


@dataclass(frozen=True)
class _Group:
    """A dashed frame: one gate-definition call, in columns and lanes."""

    label: str
    first_column: int
    last_column: int
    top_lane: int
    bottom_lane: int


@dataclass
class _Layout:
    """Where every op goes.

    ``bit_lane`` maps each classical bit to the lane it was measured on.
    """

    placements: list[tuple[LogicalOp, int]] = field(default_factory=list)
    groups: list[_Group] = field(default_factory=list)
    bit_lane: dict[int, int] = field(default_factory=dict)

    @property
    def num_columns(self) -> int:
        return 1 + max((column for _, column in self.placements), default=-1)


def _span(op: LogicalOp, bit_lane: Mapping[int, int]) -> range:
    """Lanes an op occupies, including a classical lane it depends on."""
    lanes = list(op.lanes)
    if op.condition is not None:
        lanes.append(bit_lane[op.condition[0]])
    return range(min(lanes), max(lanes) + 1)


def _layout(circuit: LogicalCircuit) -> _Layout:
    """Pack ops into columns as early as possible; keep groups contiguous."""
    layout = _Layout()
    frontier = [0] * circuit.num_qubits  # first free column per lane

    def place(op: LogicalOp) -> int:
        span = _span(op, layout.bit_lane)
        column = max(frontier[lane] for lane in span)
        layout.placements.append((op, column))
        for lane in span:
            frontier[lane] = column + 1
        if op.name == "measure":
            layout.bit_lane[op.clbit] = op.targets[0]
        return column

    ops = circuit.ops
    i = 0
    while i < len(ops):
        if ops[i].group_id is None:
            place(ops[i])
            i += 1
            continue
        j = i
        while j < len(ops) and ops[j].group_id == ops[i].group_id:
            j += 1
        run = ops[i:j]
        lanes = [lane for op in run for lane in _span(op, layout.bit_lane)]
        top, bottom = min(lanes), max(lanes)
        height = bottom + 1 - top
        start = max(frontier[top : bottom + 1])
        # Every lane of the frame starts together, so nothing slips inside.
        frontier[top : bottom + 1] = [start] * height
        end = max(place(op) for op in run)
        frontier[top : bottom + 1] = [end + 1] * height
        layout.groups.append(
            _Group(run[0].group or "", start, end, top, bottom)
        )
        i = j
    return layout


# ------------------------------------------------------------------- labels


def _width(text: str, size: float = FONT_SIZE) -> float:
    """Rendered width of ``text`` (mathtext allowed) in lane units."""
    points = TextPath((0, 0), text, size=size).get_extents().width
    return points / 72 / INCH_PER_UNIT


def _angle(value: float) -> str:
    """``π/2``-style text when the angle is a simple multiple of π."""
    ratio = Fraction(value / math.pi).limit_denominator(16)
    if ratio and math.isclose(float(ratio) * math.pi, value, abs_tol=1e-9):
        num, den = ratio.numerator, ratio.denominator
        top = {1: r"\pi", -1: r"-\pi"}.get(num, rf"{num}\pi")
        return top if den == 1 else rf"{top}/{den}"
    return f"{value:.3f}"


def _gate_label(op: LogicalOp) -> str:
    symbol = _SYMBOLS.get(op.name, rf"\mathrm{{{op.name}}}")
    if op.params:
        symbol += "(" + ", ".join(_angle(p) for p in op.params) + ")"
    return f"${symbol}$"


def _math(text: str) -> str:
    """Upright mathtext, so names match the Computer Modern of the gates."""
    return r"$\mathrm{" + text.replace("_", r"\_").replace(" ", r"\ ") + "}$"


def _is_cnot(op: LogicalOp) -> bool:
    return op.name == "x" and bool(op.controls) and op.condition is None


def _is_cz(op: LogicalOp) -> bool:
    """CZ is symmetric, so textbooks draw it as two dots."""
    return op.name == "z" and len(op.controls) == 1 and op.condition is None


def _op_width(op: LogicalOp) -> float:
    if op.name == "measure" or _is_cnot(op) or _is_cz(op):
        return BOX
    return max(BOX, _width(_gate_label(op)) + BOX_PADDING)


def _y(lane: int) -> float:
    return -float(lane)


# ----------------------------------------------------------------- drawing


def _render(circuit: LogicalCircuit, inputs: dict[str, str]) -> Figure:
    layout = _layout(circuit)
    n = circuit.num_qubits

    widths = [BOX] * layout.num_columns
    for op, column in layout.placements:
        widths[column] = max(widths[column], _op_width(op))
    pad_left = [0.0] * layout.num_columns
    pad_right = [0.0] * layout.num_columns
    for group in layout.groups:
        pad_left[group.first_column] = GROUP_PAD + 0.06
        pad_right[group.last_column] = GROUP_PAD + 0.06

    centers, x = [], GAP
    for width, left, right in zip(widths, pad_left, pad_right, strict=True):
        centers.append(x + left + width / 2)
        x += left + width + right + GAP
    x_end = x

    # Left labels, right to left: input ket, qubit index, brace, register name.
    kets = [
        "$" + inputs.get(name, DEFAULT_INPUT) + "$"
        for name, _ in circuit.qregs
    ]
    index_right = -0.08 - max(_width(k) for k in kets) - LABEL_GAP
    brace_x = index_right - 0.22
    name_lefts = [
        (brace_x - 0.24 if size > 1 else index_right) - _width(_math(name))
        for name, size in circuit.qregs
    ]
    x_min = min(name_lefts) - 0.1
    top_margin = 0.75 if layout.groups else 0.5
    bottom_margin = 0.55

    figure = Figure(
        figsize=(
            (x_end + 0.2 - x_min) * INCH_PER_UNIT,
            (n - 1 + top_margin + bottom_margin) * INCH_PER_UNIT,
        )
    )
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(x_min, x_end + 0.2)
    ax.set_ylim(-(n - 1) - bottom_margin, top_margin)
    ax.set_aspect("equal")
    ax.axis("off")

    _draw_register_labels(ax, circuit, kets, index_right, brace_x)
    _draw_wires(ax, layout, centers, n, x_end)
    _draw_groups(ax, layout, centers, widths)
    for op, column in layout.placements:
        _draw_op(ax, op, centers[column], layout.bit_lane)
    return figure


def _draw_register_labels(
    ax: Axes,
    circuit: LogicalCircuit,
    kets: list[str],
    index_right: float,
    brace_x: float,
) -> None:
    lane = 0
    for (name, size), ket in zip(circuit.qregs, kets, strict=True):
        first, last = lane, lane + size - 1
        for k in range(size):
            ax.text(
                -0.08,
                _y(lane + k),
                ket,
                ha="right",
                va="center",
                fontsize=FONT_SIZE,
            )
            if size > 1:
                ax.text(
                    index_right,
                    _y(lane + k),
                    f"${k}$",
                    ha="right",
                    va="center",
                    fontsize=SMALL_FONT,
                )
        if size > 1:
            _brace(ax, brace_x, _y(first) + 0.32, _y(last) - 0.32)
            ax.text(
                brace_x - 0.24,
                (_y(first) + _y(last)) / 2,
                _math(name),
                ha="right",
                va="center",
                fontsize=FONT_SIZE,
            )
        else:
            ax.text(
                index_right,
                _y(lane),
                _math(name),
                ha="right",
                va="center",
                fontsize=FONT_SIZE,
            )
        lane += size


def _draw_wires(
    ax: Axes, layout: _Layout, centers: list[float], n: int, x_end: float
) -> None:
    """Single (quantum) wire up to a measurement, double (classical) after."""
    classical_from = {
        op.targets[0]: centers[column] + BOX / 2
        for op, column in layout.placements
        if op.name == "measure"
    }
    for lane in range(n):
        split = classical_from.get(lane, x_end)
        ax.plot([0.0, split], [_y(lane)] * 2, color="black", lw=WIRE, zorder=0)
        if split < x_end:
            for offset in (-DOUBLE, DOUBLE):
                ax.plot(
                    [split, x_end],
                    [_y(lane) + offset] * 2,
                    color="black",
                    lw=CLASSICAL,
                    zorder=0,
                )


def _draw_groups(
    ax: Axes, layout: _Layout, centers: list[float], widths: list[float]
) -> None:
    """Dashed frame and name around each gate-definition call."""
    for group in layout.groups:
        left = (
            centers[group.first_column]
            - widths[group.first_column] / 2
            - GROUP_PAD
        )
        right = (
            centers[group.last_column]
            + widths[group.last_column] / 2
            + GROUP_PAD
        )
        upper = _y(group.top_lane) + BOX / 2 + GROUP_PAD
        lower = _y(group.bottom_lane) - BOX / 2 - GROUP_PAD
        ax.add_patch(
            Rectangle(
                (left, lower),
                right - left,
                upper - lower,
                fill=False,
                lw=0.7,
                ls=(0, (3, 2)),
                edgecolor="black",
                zorder=1,
            )
        )
        ax.text(
            left,
            upper + 0.05,
            _math(group.label.replace("_", " ")),
            ha="left",
            va="bottom",
            fontsize=SMALL_FONT,
        )


def _brace(
    ax: Axes, x: float, top: float, bottom: float, width: float = 0.16
) -> None:
    """A left-pointing curly brace from ``top`` to ``bottom`` at ``x``."""
    middle = (top + bottom) / 2
    r = min(0.14, (top - bottom) / 4)  # corner radius
    h = x - width / 2  # the brace's vertical stroke
    vertices = [
        (x, top),
        (h, top),
        (h, top - r),
        (h, middle + r),
        (h, middle),
        (x - width, middle),
        (h, middle),
        (h, middle - r),
        (h, bottom + r),
        (h, bottom),
        (x, bottom),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
    ]
    ax.add_patch(
        PathPatch(Path(vertices, codes), fill=False, lw=0.8, edgecolor="black")
    )


def _draw_op(
    ax: Axes, op: LogicalOp, center: float, bit_lane: Mapping[int, int]
) -> None:
    if op.name == "measure":
        _draw_meter(ax, center, _y(op.targets[0]))
        return

    target = _y(op.targets[0])
    quantum = [_y(lane) for lane in op.lanes]
    if len(quantum) > 1:
        ax.plot(
            [center, center],
            [min(quantum), max(quantum)],
            color="black",
            lw=WIRE,
            zorder=2,
        )
    for lane, filled in op.controls:
        ax.add_patch(
            Circle(
                (center, _y(lane)),
                DOT,
                facecolor="black" if filled else "white",
                edgecolor="black",
                lw=WIRE,
                zorder=4,
            )
        )

    if op.condition is not None:  # classical double wire from the measured bit
        source = _y(bit_lane[op.condition[0]])
        edge = target + (BOX / 2 if source > target else -BOX / 2)
        for offset in (-DOUBLE, DOUBLE):
            ax.plot(
                [center + offset] * 2,
                [source, edge],
                color="black",
                lw=CLASSICAL,
                zorder=2,
            )
        ax.add_patch(
            Circle((center, source), DOT * 0.8, color="black", zorder=4)
        )

    if _is_cz(op):
        ax.add_patch(Circle((center, target), DOT, color="black", zorder=4))
    elif _is_cnot(op):
        ax.add_patch(
            Circle(
                (center, target),
                TARGET,
                facecolor="white",
                edgecolor="black",
                lw=WIRE,
                zorder=3,
            )
        )
        ax.plot(
            [center - TARGET, center + TARGET],
            [target] * 2,
            color="black",
            lw=WIRE,
            zorder=4,
        )
        ax.plot(
            [center] * 2,
            [target - TARGET, target + TARGET],
            color="black",
            lw=WIRE,
            zorder=4,
        )
    else:
        width = _op_width(op)
        ax.add_patch(
            Rectangle(
                (center - width / 2, target - BOX / 2),
                width,
                BOX,
                facecolor="white",
                edgecolor="black",
                lw=WIRE,
                zorder=3,
            )
        )
        ax.text(
            center,
            target,
            _gate_label(op),
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
            zorder=4,
        )


def _draw_meter(ax: Axes, center: float, y: float) -> None:
    ax.add_patch(
        Rectangle(
            (center - BOX / 2, y - BOX / 2),
            BOX,
            BOX,
            facecolor="white",
            edgecolor="black",
            lw=WIRE,
            zorder=3,
        )
    )
    ax.add_patch(
        Arc(
            (center, y - 0.12),
            0.38,
            0.36,
            theta1=0,
            theta2=180,
            color="black",
            lw=0.8,
            zorder=4,
        )
    )
    ax.annotate(
        "",
        xy=(center + 0.16, y + 0.17),
        xytext=(center, y - 0.12),
        arrowprops=dict(
            arrowstyle="-|>", lw=0.8, color="black", mutation_scale=6
        ),
        zorder=4,
    )
