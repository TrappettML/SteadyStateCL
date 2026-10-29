"""Figure standard (PLAN.md §12.0, §12.7): the only style file.

Inspection figures: PNG only, sized by panel (2.6 in × 2.0 in) so every
panel is legible at 100 % zoom, one style for every figure of
``analyze.py``, ``plot_norms.py``, ``plot_check2.py`` and ``validate.py``.
No script sets ``rcParams``, picks a colour, or names a quantity itself;
everything comes from here.
"""
from __future__ import annotations

import logging
import os
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter, NullLocator  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# colour (bound to the readout, never to rank)
# ---------------------------------------------------------------------------
READOUT_COLOR = {
    "gated_shared": "#0072B2", "learned_shared": "#56B4E9",
    "gated_frozen": "#D55E00", "frozen": "#E69F00",
    "task_heads": "#009E73",
}
READOUT_LABEL = {
    "gated_shared": "gated shared", "learned_shared": "learned shared",
    "gated_frozen": "gated frozen", "frozen": "frozen", "task_heads": "task heads",
}
#: ungated conditions dashed, so gating is visible without colour
READOUT_STYLE = {
    "gated_shared": "-", "learned_shared": "--", "gated_frozen": "-", "frozen": "--", "task_heads": "-",
}
#: one sequential blue for width N when it is a trace rather than a column (light → dark)
N_SHADES = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
GREY_REF, GREY_BURNIN, GREY_CAPTION, GREY_NA, RED_FLAG = "0.4", "0.92", "0.3", "0.85", "#CC0000"
#: white → red for the flagged fraction (§12.6 C2-E flag table)
FLAG_CMAP = LinearSegmentedColormap.from_list("flag", ["white", RED_FLAG])
#: line widths of PLAN.md §12.0 (layering bottom → top)
LW_REF, LW_OVERLAY, LW_TRACE, LW_MEAN, LW_EVENT = 0.8, 0.6, 1.6, 2.2, 0.8
ALPHA_TASK, ALPHA_SEED, ALPHA_BAND = 0.08, 0.5, 0.2
PANEL = (2.6, 2.0)          # inches, w × h
MAX_COLS = 6
FS_MIN = 8
MIN_BOX_N = 10              # a box summarises a distribution only from this many values
#: inches reserved above the panels: suptitle, one caption line, one legend row (§12.0 Legend)
_TITLE_IN, _CAPTION_LINE_IN, _LEGEND_IN, _PAD_IN = 0.28, 0.16, 0.30, 0.06
_BOTTOM_IN = 1.3            # room under the panels for tick labels and x-labels
_CHAR_IN = 0.07             # ≈ width of one 9-pt DejaVu Sans character

#: display names of every per-task scalar, trace quantity and derived statistic (PLAN.md §12.7 G2);
#: scripts never show a schema key
QUANTITY_LABEL = {
    "Q_act_pre": r"$Q_{ii}$ (active, pre)", "Q_act_post": r"$Q_{ii}$ (active)", "Q_all_pre": r"$Q_{ii}$ (all, pre)",
    "Q_all_post": r"$Q_{ii}$ (all)", "W_fro2_pre": r"$\|W\|_F^2/D$ (pre)", "W_fro2_post": r"$\|W\|_F^2/D$",
    "a2_act_pre": r"$a_i^2$ (active, pre)", "a2_act_post": r"$a_i^2$ (active)", "a2_all_pre": r"$a_i^2$ (all, pre)",
    "a2_all_post": r"$a_i^2$ (all)",
    "aQ_act_pre": r"mean $|a_i|\sqrt{Q_{ii}}$ (pre)", "aQ_act_post": r"mean $|a_i|\sqrt{Q_{ii}}$",
    "aQ_all_pre": r"mean $|a_i|\sqrt{Q_{ii}}$ (all, pre)", "aQ_all_post": r"mean $|a_i|\sqrt{Q_{ii}}$ (all)",
    "aQ_max_pre": r"max $|a_i|\sqrt{Q_{ii}}$ (pre)", "aQ_max_post": r"max $|a_i|\sqrt{Q_{ii}}$",
    "heads_fro2_pre": r"mean $|a_k|^2$ (heads, pre)", "heads_fro2_post": r"mean $|a_k|^2$ (heads)",
    "logf_sum": r"$\Sigma\,\log f$", "logf_max": r"max $|\log f|$",
    "Q_q05": r"$Q_{ii}$ 5 %", "Q_q50": r"$Q_{ii}$ median", "Q_q95": r"$Q_{ii}$ 95 %",
    "aQ_q05": r"$|a_i|\sqrt{Q_{ii}}$ 5 %", "aQ_q95": r"$|a_i|\sqrt{Q_{ii}}$ 95 %",
    "S0": "S(0)", "E0": "E(0)", "L_end": r"$L$ (end of task)", "abar2": r"$\bar a^2$", "n_active": "active units",
    "dm_half": r"$\Delta m_{1/2}/m_{1/2}$",
    # trace quantities (protocol.TRACE_QUANT)
    "a2_act": r"$a_i^2$ (active)", "aQ_act": r"mean $|a_i|\sqrt{Q_{ii}}$", "Q_all": r"$Q_{ii}$ (all)",
    "logf": r"$\log f$", "loss": "loss",
    "Q_act_pre-Q_act_post": r"$Q_{ii}$ pre − post",
}


def qlabel(key: str) -> str:
    """Display label of a quantity key (``QUANTITY_LABEL``); a key without one is an error, not a fallback."""
    return QUANTITY_LABEL[key]


def fmt(v) -> str:
    """Parameter value as text: ``4`` not ``4.0``, ``0.5`` not ``0.50``; a numeric string is formatted too, other strings pass through."""
    if isinstance(v, str):
        try:
            return f"{float(v):g}"
        except ValueError:
            return v
    return f"{float(v):g}"


def fmt2(v) -> str:
    """Figure-level number to 2 significant figures (a colour limit, a fitted slope)."""
    if not np.isfinite(v):
        return "n/a"
    return f"{float(v):.2g}"


def apply_style() -> None:
    """Set the rcParams of PLAN.md §12.0 (called once at import)."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.titlesize": 10, "figure.titlesize": 11, "figure.titleweight": "bold",
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.which": "major", "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.formatter.useoffset": False, "axes.xmargin": 0.05, "axes.ymargin": 0.05,
        "legend.frameon": False, "lines.linewidth": LW_TRACE, "lines.markersize": 4.5,
        "savefig.dpi": 150, "figure.dpi": 100,
        "hatch.linewidth": 0.6,
    })


apply_style()


# ---------------------------------------------------------------------------
# figure construction and saving
# ---------------------------------------------------------------------------
def _title_in(fig) -> float:
    """Inches of the suptitle band (grows with a wrapped title)."""
    return _TITLE_IN + 0.19 * (fig._mup.get("title_lines", 1) - 1)


def _top_in(fig) -> float:
    """Inches reserved above the panels for the current title lines, caption lines and legend rows."""
    return _title_in(fig) + _CAPTION_LINE_IN * fig._mup["caption_lines"] + _LEGEND_IN * fig._mup.get("legend_rows", 1) + _PAD_IN


def _relayout(fig) -> None:
    """Re-reserve the top band after the caption line count changed (grows the figure, never the panels shrink)."""
    w, h = fig.get_size_inches()
    top = _top_in(fig)
    h_new = fig._mup["panel_h"] + _BOTTOM_IN + top
    if abs(h_new - h) > 1e-6:
        fig.set_size_inches(w, h_new)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - top / h_new))
    if fig._suptitle is not None:
        fig._suptitle.set_y(1 - 0.06 / h_new)
    if fig._mup.get("caption") is not None:
        fig._mup["caption"].set_y(1 - _title_in(fig) / h_new)
    if fig._mup.get("legend") is not None:
        y = 1 - (_title_in(fig) + _CAPTION_LINE_IN * fig._mup["caption_lines"] + _PAD_IN) / h_new
        fig._mup["legend"].set_bbox_to_anchor((0.5, y))


def grid(nrows: int, ncols: int, sharex="col", sharey="row", panel=PANEL, extra_w: float = 0.0, **kw):
    """``plt.subplots`` at a fixed physical size per panel, constrained layout, ≤ 4 ticks per linear axis.

    The top band (suptitle, caption, one legend row) is reserved so nothing
    drawn there overlaps the panels (§12.0 Legend).

    Parameters
    ----------
    nrows, ncols : int
        Figures wider than ``MAX_COLS`` panels should be split (see ``col_chunks``);
        a warning is issued here and again by ``save``.
    sharex, sharey : bool or {"col", "row", "all"}
    panel : (w, h) inches of one panel
    extra_w : float
        Extra figure width in inches (room for a colorbar).
    **kw
        Passed to ``plt.subplots``.

    Returns
    -------
    (fig, axes)
        ``axes`` always a 2-D array (``squeeze=False``).
    """
    if ncols > MAX_COLS:
        warnings.warn(f"figure has {ncols} columns (> {MAX_COLS}); split it instead of shrinking panels")
    w, h = panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols + 1.4 + extra_w, h * nrows + _BOTTOM_IN),
                             sharex=sharex, sharey=sharey, squeeze=False, layout="constrained", **kw)
    fig._mup = {"caption_lines": 1, "panel_h": h * nrows, "caption": None, "legend": None}
    _relayout(fig)
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    return fig, axes


def col_chunks(items, n: int = MAX_COLS) -> list:
    """Split a column list into chunks of at most ``n`` (one figure per chunk)."""
    items = list(items)
    return [items[i:i + n] for i in range(0, len(items), n)] or [[]]


def logaxis(ax, axis: str = "x") -> None:
    """Log scale with ``LogLocator(numticks=4)`` and plain ``10^k`` labels.

    Parameters
    ----------
    ax : matplotlib Axes
    axis : {"x", "y"}
    """
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(LogLocator(base=10, numticks=4))
    a.set_major_formatter(LogFormatterMathtext())
    a.set_minor_formatter(NullFormatter())


def discrete_ticks(ax, values, axis: str = "x") -> None:
    """Ticks at exactly the given values with plain labels (an axis carrying ≤ 6 discrete values, §12.0 Text).

    Works on linear and log axes alike; minor ticks are removed.

    Parameters
    ----------
    ax : matplotlib Axes
    values : sequence of numbers
    axis : {"x", "y"}
    """
    a = ax.xaxis if axis == "x" else ax.yaxis
    vals = sorted({float(v) for v in values})
    a.set_major_locator(FixedLocator(vals))
    a.set_major_formatter(FuncFormatter(lambda v, _pos: fmt(v)))
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def age_axis(ax, max_age: float) -> None:
    """Age axis with age 0 plotted: ``symlog`` with ``linthresh 1``, ticks at 0 and the decades (§12.7 G13).

    Parameters
    ----------
    ax : matplotlib Axes
    max_age : float
        Largest age plotted (sets the tick list).
    """
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.3)
    ticks = [0.0] + [10.0 ** k for k in range(0, 5) if 10.0 ** k <= max(max_age, 1.0)]
    discrete_ticks(ax, ticks, "x")
    ax.set_xlim(-0.3, max(max_age, 1.0) * 1.3)


def lifetime_axis(ax, max_x: float, min_pos: float) -> None:
    """Age in lifetimes (m κ_s²/N) with age 0 plotted: ``symlog`` whose linear part ends at the smallest positive value.

    Parameters
    ----------
    ax : matplotlib Axes
    max_x : float
        Largest value plotted.
    min_pos : float
        Smallest positive value plotted (age 1 of the largest N).
    """
    lt = max(min_pos, 1e-6)
    ax.set_xscale("symlog", linthresh=lt, linscale=0.3)
    ticks = [0.0] + [10.0 ** k for k in range(-4, 5) if lt <= 10.0 ** k <= max(max_x, lt) * 1.01]
    discrete_ticks(ax, ticks[:6], "x")
    ax.set_xlim(-0.3 * lt, max(max_x, lt) * 1.3)


MARK_GLYPH = {"ok": "✓", "warn": "~", "fail": "✗", "na": "—"}
MARK_COLOR = {"ok": "k", "warn": GREY_REF, "fail": RED_FLAG, "na": GREY_CAPTION}


def check_table(ax, marks: dict, rows: list, cols: list) -> None:
    """The ✓/✗ checks table of PLAN.md §12.6: rows = checks, columns = readout × N grouped by readout.

    Parameters
    ----------
    ax : matplotlib Axes
    marks : dict
        ``{(row_key, (readout, N)): (mark, text)}`` with ``mark`` one of
        ``"ok"``, ``"warn"``, ``"fail"``, ``"na"`` or ``None`` (text only, grey)
        and ``text`` the number the mark was judged on (printed at 8 pt).
    rows : list of (key, label)
    cols : list of (readout, N), grouped by readout in the fixed order
    """
    nr, nc = len(rows), len(cols)
    for j, (ro, N) in enumerate(cols):
        for i, (rk, _) in enumerate(rows):
            mark, text = marks.get((rk, (ro, N)), (None, ""))
            if mark == "fail":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=RED_FLAG, alpha=0.12, lw=0, zorder=1))
            elif mark == "na":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, hatch="///", fc="white", ec=GREY_NA, lw=0, zorder=1))
            if mark is not None:
                ax.text(j, i - 0.14, MARK_GLYPH[mark], ha="center", va="center", fontsize=11, color=MARK_COLOR[mark], zorder=3,
                        fontweight="bold" if mark == "fail" else "normal")
            if text:
                ax.text(j, i + (0.27 if mark is not None else 0.0), text, ha="center", va="center", fontsize=FS_MIN,
                        color=GREY_CAPTION if mark in (None, "na", "warn") else "k", zorder=3)
    # header: N per column, readout once per group in its colour, thin rules between groups
    for j, (ro, N) in enumerate(cols):
        ax.text(j, -0.75, f"N = {fmt(N)}", ha="center", va="center", fontsize=FS_MIN, color="k")
    j0 = 0
    while j0 < nc:
        j1 = j0
        while j1 + 1 < nc and cols[j1 + 1][0] == cols[j0][0]:
            j1 += 1
        ro = cols[j0][0]
        ax.text((j0 + j1) / 2, -1.45, READOUT_LABEL[ro], ha="center", va="center", fontsize=9, color=READOUT_COLOR[ro], fontweight="bold")
        if j1 + 1 < nc:
            ax.axvline(j1 + 0.5, color=GREY_REF, lw=LW_REF, zorder=2)
        j0 = j1 + 1
    for i in range(nr + 1):
        ax.axhline(i - 0.5, color=GREY_NA, lw=0.5, zorder=2)
    ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(nr - 0.5, -1.9)
    ax.set_xticks([])
    ax.set_yticks(range(nr)); ax.set_yticklabels([lab for _, lab in rows], fontsize=9)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax._mup_no_const_check = True
    ax._mup_narrow_ok = True


def sym_limits(axes, values, floor: float, clip: float | None = None, axis: str = "y") -> float:
    """One symmetric range for every axes in ``axes`` from the pooled finite values (§12.7 G6).

    Parameters
    ----------
    axes : iterable of Axes
    values : array-like
        Everything plotted on those axes.
    floor : float
        The range is at least ±floor (so a threshold band stays visible).
    clip : float, optional
        The range is at most ±clip; callers draw out-of-range values at the edge (``edge_markers``).
    axis : {"y", "x"}

    Returns
    -------
    float
        The half-range applied.
    """
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    lim = max(float(floor), float(np.abs(v).max()) * 1.08) if len(v) else float(floor)
    if clip is not None:
        lim = min(lim, float(clip))
    for ax in np.ravel(axes):
        (ax.set_ylim if axis == "y" else ax.set_xlim)(-lim, lim)
    return lim


def _text_fits(fig, t) -> bool:
    """Whether a Text's rendered box lies inside the figure canvas."""
    try:
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    except Exception:
        return True
    fb = fig.bbox
    return bb.x0 >= fb.x0 - 1 and bb.x1 <= fb.x1 + 1 and bb.y0 >= fb.y0 - 1 and bb.y1 <= fb.y1 + 1


def _drawn_texts(fig) -> list:
    """Every Text that will actually be drawn: figure texts, titles, axis labels, legends and the tick labels in view."""
    out = list(fig.texts)
    if fig._suptitle is not None:
        out.append(fig._suptitle)
    for leg in fig.legends:
        out += list(leg.get_texts())
    for ax in fig.axes:
        out += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        if ax.get_legend() is not None:
            out += list(ax.get_legend().get_texts())
        for axis in (ax.xaxis, ax.yaxis):
            ticks = axis._update_ticks() if hasattr(axis, "_update_ticks") else axis.get_major_ticks()
            for tick in ticks:
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible():
                        out.append(lab)
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def _check(fig) -> None:
    """Figure-level checks of PLAN.md §12.0; each failure is a warning."""
    if fig._suptitle is None:
        warnings.warn("figure without a suptitle")
    panels = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
    cols = {round(ax.get_position().x0, 3) for ax in panels}
    if len(cols) > MAX_COLS:
        warnings.warn(f"figure is {len(cols)} panels wide (> {MAX_COLS})")
    w_in = fig.get_size_inches()[0]
    for ax in panels:
        if ax.get_position().width * w_in < 2.0 - 1e-6 and not getattr(ax, "_mup_narrow_ok", False):
            warnings.warn("panel narrower than 2 in")
            break
    for ax in fig.axes:
        for a, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            if isinstance(a.get_major_locator(), FixedLocator):      # discrete / categorical axis
                continue
            ticks = [t for t in a.get_majorticklocs() if min(lim) <= t <= max(lim)]
            if len(ticks) > 6:
                warnings.warn(f"axis with {len(ticks)} ticks (> 6)")
                break
    texts = _drawn_texts(fig)
    for t in texts:
        if t.get_fontsize() < FS_MIN:
            warnings.warn(f"text smaller than {FS_MIN} pt: {t.get_text()[:30]!r}")
            break
    for t in texts:
        if not _text_fits(fig, t):
            warnings.warn(f"text extends past the figure edge: {t.get_text()[:30]!r}")
            break
    for ax in panels:
        if ax.get_legend() is not None:
            warnings.warn("legend inside an axes (use fig_legend)")
            break
    for ax in panels:
        if getattr(ax, "_mup_na", False) or getattr(ax, "_mup_no_const_check", False):
            continue
        ys = [np.asarray(l.get_ydata(), float) for l in ax.get_lines()
              if l.get_zorder() >= 2 and len(l.get_xdata()) >= 3]
        ys = [y[np.isfinite(y)] for y in ys]
        ys = [y for y in ys if len(y) >= 3]
        if ys and all(np.ptp(y) == 0 for y in ys):
            warnings.warn("panel whose plotted values are all identical (should be an n/a slot)")
            break


def save(fig, path: str, figdata: dict | None = None, index: list | None = None,
         family: str = "", cell: str = "") -> str:
    """Save ``<path>.png`` (never ``bbox_inches="tight"``), run the §12.0 checks and close the figure.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str
        Stem without extension; directories are created.
    figdata : dict, optional
        Arrays actually plotted; written to ``<path>.npz`` when given (``--save-figdata``).
    index : list, optional
        ``INDEX.md`` rows; ``(family, cell, file)`` is appended.
    family, cell : str

    Returns
    -------
    str
        The PNG path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for ax in fig.axes:
        for axis_name, a in (("x", ax.xaxis), ("y", ax.yaxis)):
            if a.get_scale() == "log" and not isinstance(a.get_major_locator(), (LogLocator, FixedLocator)):
                logaxis(ax, axis_name)
    fig.canvas.draw()          # final tick positions and text extents for the checks
    _check(fig)
    out = path + ".png"
    fig.savefig(out)
    plt.close(fig)
    if figdata:
        np.savez_compressed(path + ".npz", **{k: np.asarray(v) for k, v in figdata.items()})
    if index is not None:
        index.append((family, cell, os.path.relpath(out, os.path.dirname(os.path.dirname(path)))))
    return out


def write_index(fig_dir: str, index: list) -> None:
    """Write ``figures/INDEX.md``: one line per figure (family, cell, file).

    Parameters
    ----------
    fig_dir : str
    index : list of (family, cell, file)
    """
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, "INDEX.md"), "w") as f:
        f.write("| family | cell | file |\n|---|---|---|\n")
        for fam, cell, path in sorted(index):
            f.write(f"| {fam} | {cell} | `{path}` |\n")


# ---------------------------------------------------------------------------
# titles, captions, names, legends
# ---------------------------------------------------------------------------
_GREEK = {"κ": r"$\kappa$", "δ₀": r"$\delta_0$", "λ": r"$\lambda$", "η": r"$\eta$", "ρ": r"$\rho$", "β": r"$\beta$"}


def mathify(s: str) -> str:
    """Replace Greek letters in a label by mathtext (fonts lack the subscript glyph)."""
    for k, v in _GREEK.items():
        s = s.replace(k, v)
    return s


def cell_title(fig, kappa=None, d0=None, lam=None, solver=None, eta=None, extra: str = "") -> None:
    """One bold ``suptitle`` stating the cell in words, e.g. ``κ = 1, δ₀ = 4, λ = 1, population, η = 1``.

    Parameters
    ----------
    fig : matplotlib Figure
    kappa, d0, lam, solver, eta : values or None
        Fields fixed in the figure; ``None`` is omitted.  Numbers print through ``fmt``.
    extra : str
        Appended after the cell (e.g. a readout name).
    """
    parts = []
    if kappa is not None:
        parts.append(rf"$\kappa$ = {'√N' if kappa == 'sqrtN' else fmt(kappa)}")
    if d0 is not None:
        parts.append(rf"$\delta_0$ = {fmt(d0)}")
    if lam is not None:
        parts.append(rf"$\lambda$ = {fmt(lam)}")
    if solver is not None:
        parts.append({"pop": "population", "sgd": "SGD"}.get(solver, solver) + (rf", $\eta$ = {fmt(eta)}" if eta is not None else ""))
    if extra:
        parts.append(mathify(extra))
    text = ", ".join(parts) if parts else mathify(extra) or " "
    w, h = fig.get_size_inches()
    # wrap at a comma when the bold 11-pt title would overrun the figure (≈ 0.095 in per character)
    plain = text.replace("$", "").replace("\\", "")
    if len(plain) * 0.095 > w - 0.3 and len(parts) > 1:
        cut = max(1, len(parts) // 2)
        text = ", ".join(parts[:cut]) + "\n" + ", ".join(parts[cut:])
    fig._mup["title_lines"] = text.count("\n") + 1
    fig.suptitle(text, y=1 - 0.06 / h, va="top", linespacing=1.1)
    _relayout(fig)


def caption(fig, text: str) -> None:
    """One caption under the suptitle (9 pt, grey) saying what overlays and bands are; wraps to the figure width (§12.7 G11).

    Parameters
    ----------
    fig : matplotlib Figure
    text : str
    """
    w = fig.get_size_inches()[0]
    lines = textwrap.wrap(text, width=max(20, int((w - 0.4) / _CHAR_IN)))
    if fig._mup.get("caption") is not None:
        fig._mup["caption"].remove()
    fig._mup["caption_lines"] = max(1, len(lines))
    h = fig.get_size_inches()[1]
    fig._mup["caption"] = fig.text(0.5, 1 - _title_in(fig) / h, mathify("\n".join(lines)), ha="center", va="top",
                                   fontsize=9, color=GREY_CAPTION, linespacing=1.15)
    _relayout(fig)


def cell_stem(kappa=None, d0=None, lam=None, solver=None, eta=None, readout=None) -> str:
    """File-name stem ``k{κ}_d{δ₀}_lam{λ}_{solver}_eta{η}`` (PLAN.md §12.4); ``None`` fields are dropped.

    ``readout`` goes first when given (figures for a single readout).
    """
    parts = []
    if readout is not None:
        parts.append(str(readout))
    if kappa is not None:
        parts.append(f"k{kappa}")
    if d0 is not None:
        parts.append(f"d{d0}")
    if lam is not None:
        parts.append(f"lam{lam}")
    if solver is not None:
        parts.append(str(solver))
    if eta is not None:
        parts.append(f"eta{eta}")
    return "_".join(parts)


def readout_handles(readouts=None, solid: bool = False, marker: bool = False) -> list:
    """Legend handles for readouts in the fixed order of PLAN.md §4.

    Parameters
    ----------
    readouts : iterable of str, optional
        Subset (default: all five).
    solid : bool
        Draw every handle solid (when line style means something else in the figure).
    marker : bool
        Marker handles (hollow for ungated) instead of lines.
    """
    from mup.config import is_gated
    ros = [r for r in READOUT_COLOR if readouts is None or r in set(readouts)]
    out = []
    for r in ros:
        col = READOUT_COLOR[r]
        if marker:
            out.append(Line2D([], [], color=col, ls="none", marker="o", ms=5, mfc=col if is_gated(r) else "white", label=READOUT_LABEL[r]))
        else:
            out.append(Line2D([], [], color=col, ls="-" if solid else READOUT_STYLE[r], lw=LW_TRACE, label=READOUT_LABEL[r]))
    return out


def fig_legend(fig, readouts=None, styles=None, handles=None, solid: bool = False, marker: bool = False):
    """The figure's one legend: a single row above the panels, below the caption (§12.0 Legend, §12.7 G1).

    Parameters
    ----------
    fig : matplotlib Figure
    readouts : iterable of str, optional
        Readouts to list (fixed order); ``None`` with ``handles`` given lists only ``handles``.
    styles : list of (linestyle, label), optional
        Extra handles when line style means something other than gating (e.g. half 0 / half 1);
        the readout handles are then drawn solid.
    handles : list of Line2D, optional
        Extra or replacement handles (e.g. N shades).
    solid, marker : bool
        Passed to ``readout_handles``.

    Returns
    -------
    matplotlib Legend
    """
    hs = [] if (readouts is None and handles is not None) else readout_handles(readouts, solid or bool(styles), marker)
    for ls, lab in (styles or []):
        hs.append(Line2D([], [], color="k", ls=ls, lw=1.2, label=lab))
    hs += list(handles or [])
    if fig._mup.get("legend") is not None:
        fig._mup["legend"].remove()
    # wrap into rows when the handles do not fit the figure width (handle ≈ 0.35 in + text + spacing)
    w = fig.get_size_inches()[0]
    widths = [0.35 + _CHAR_IN * len(h.get_label()) + 0.25 for h in hs]
    ncol = max(1, len(hs))
    while ncol > 1 and sum(sorted(widths, reverse=True)[:ncol]) > w - 0.4:
        ncol -= 1
    fig._mup["legend_rows"] = int(np.ceil(len(hs) / ncol)) if hs else 1
    leg = fig.legend(handles=hs, loc="upper center", ncol=ncol, frameon=False, fontsize=9,
                     handlelength=2.2, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.3)
    fig._mup["legend"] = leg
    _relayout(fig)
    return leg


def label_cols_N(axes, Ns) -> None:
    """Title the top row of a panel grid ``N = …``.

    Parameters
    ----------
    axes : 2-D array of Axes
    Ns : sequence of int
    """
    for ax, N in zip(np.atleast_2d(axes)[0], Ns):
        ax.set_title(f"N = {fmt(N)}")


def n_shade(i: int, n: int) -> str:
    """Colour for the i-th of n widths (light for small N, dark for large)."""
    return N_SHADES[int(round(i * (len(N_SHADES) - 1) / max(n - 1, 1)))]


def n_handles(Ns) -> list:
    """Legend handles for width traces in ``N_SHADES``."""
    return [Line2D([], [], color=n_shade(i, len(Ns)), lw=LW_TRACE, label=f"N = {fmt(N)}") for i, N in enumerate(Ns)]


def row_rule(fig, axes, row: int) -> None:
    """Thin grey rule across the figure between panel rows ``row − 1`` and ``row`` (drawn after layout)."""
    fig.canvas.draw()
    above = axes[row - 1, 0].get_position()
    below = axes[row, 0].get_position()
    y = (above.y0 + below.y1) / 2
    x0 = min(ax.get_position().x0 for ax in axes[row])
    x1 = max(ax.get_position().x1 for ax in axes[row])
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=GREY_REF, lw=LW_REF))


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def plot_readout(ax, x, y, ro: str, **kw):
    """Plot a trace in the readout's colour and line style (``lw`` defaults to the main-trace width).

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : array-like
    ro : str
        Readout key.
    **kw
        Passed to ``ax.plot``; may override ``color``, ``ls``, ``lw``.

    Returns
    -------
    list of Line2D
    """
    kw.setdefault("color", READOUT_COLOR[ro])
    kw.setdefault("ls", READOUT_STYLE[ro])
    kw.setdefault("lw", LW_TRACE)
    kw.setdefault("zorder", 4)
    return ax.plot(np.asarray(x, float), np.asarray(y, float), **kw)


def band(ax, x, lo, hi, color: str, alpha: float = ALPHA_BAND, **kw) -> None:
    """``fill_between`` with no edge, skipping entries where ``lo`` or ``hi`` is NaN.

    Parameters
    ----------
    ax : matplotlib Axes
    x, lo, hi : array-like
    color : str
    alpha : float
    """
    x, lo, hi = (np.asarray(v, float) for v in (x, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=alpha, lw=0, zorder=3, **kw)


def ref_line(ax, value: float = 0.0, axis: str = "y") -> None:
    """Reference line (ratio = 1, slope = 0, …) in grey, drawn below the data.

    Parameters
    ----------
    ax : matplotlib Axes
    value : float
    axis : {"y", "x"}
    """
    (ax.axhline if axis == "y" else ax.axvline)(value, color=GREY_REF, lw=LW_REF, zorder=1)


def event_line(ax, x: float) -> None:
    """Dotted black vertical line marking an event (the half boundary)."""
    ax.axvline(x, color="k", ls=":", lw=LW_EVENT, zorder=1)


def shade_burnin(ax, T_burn: int, T_meas: int, scale: float = 1.0) -> None:
    """Grey the burn-in window and mark the half boundary of the measurement window.

    Parameters
    ----------
    ax : matplotlib Axes
    T_burn, T_meas : int
    scale : float
        Multiplies task indices (κ²/N to draw the axis in lifetimes).
    """
    ax.axvspan(0, T_burn * scale, color=GREY_BURNIN, lw=0, zorder=0)
    event_line(ax, (T_burn + T_meas / 2) * scale)


def zero_band(ax, lo: float, hi: float, axis: str = "y") -> None:
    """Grey band (``GREY_BURNIN``) between ``lo`` and ``hi`` plus a grey zero line (a threshold, §12.6 C2-E).

    Parameters
    ----------
    ax : matplotlib Axes
    lo, hi : float
    axis : {"y", "x"}
    """
    (ax.axhspan if axis == "y" else ax.axvspan)(lo, hi, color=GREY_BURNIN, lw=0, zorder=0)
    ref_line(ax, 0.0, axis)


def na_slot(ax, reason: str, x: float | None = None, width: float = 0.8) -> None:
    """Hatch a not-applicable slot and print why (§12.7 G4).

    Parameters
    ----------
    ax : matplotlib Axes
    reason : str
        Printed at 8 pt (rotated when the slot is a column).
    x : float, optional
        Data x of a column slot; ``None`` hatches the whole panel.
    width : float
        Column slot width in data units.
    """
    if x is None:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, hatch="///", fc="white", ec=GREY_NA, lw=0, zorder=0.5))
        ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", fontsize=FS_MIN, color=GREY_CAPTION, zorder=6)
        ax.grid(False)
        ax._mup_na = True
    else:
        ax.axvspan(x - width / 2, x + width / 2, hatch="///", fc="white", ec=GREY_NA, lw=0, zorder=0.5)
        ax.text(x, 0.5, reason, transform=ax.get_xaxis_transform(), ha="center", va="center", rotation=90,
                fontsize=FS_MIN, color=GREY_CAPTION, zorder=6)


def flag_mark(ax, x: float, y: float, width: float = 0.8) -> None:
    """Mark a flagged cell: light red strip behind it and a red ring round its point (§12.7 G9); the marker keeps its colour.

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : float
    width : float
        Strip width in data units.
    """
    ax.axvspan(x - width / 2, x + width / 2, color=RED_FLAG, alpha=0.12, lw=0, zorder=0.6)
    if np.isfinite(y):
        ax.plot([x], [y], "o", ms=9, mfc="none", mec=RED_FLAG, mew=1.0, zorder=6, scaley=False)


def edge_markers(ax, x, y, axis: str = "y") -> None:
    """Draw ▲/▼ at the axis edge for values beyond the current limits, with the value printed (§12.7 G6, G14).

    Call after the limits are final.

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : array-like
    axis : {"y"}
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo, hi = ax.get_ylim()
    pad = 0.04 * (hi - lo)
    for xi, yi in zip(x, y):
        if not np.isfinite(yi):
            continue
        if yi > hi:
            ax.plot([xi], [hi - pad], "^", color=GREY_CAPTION, ms=5, zorder=6, scaley=False, clip_on=False)
            ax.text(xi, hi - 2.2 * pad, fmt2(yi), ha="center", va="top", fontsize=FS_MIN, color=GREY_CAPTION, zorder=6)
        elif yi < lo:
            ax.plot([xi], [lo + pad], "v", color=GREY_CAPTION, ms=5, zorder=6, scaley=False, clip_on=False)
            ax.text(xi, lo + 2.2 * pad, fmt2(yi), ha="center", va="bottom", fontsize=FS_MIN, color=GREY_CAPTION, zorder=6)


def block_mean(x, Y, w: int, cuts=()):
    """Mean over blocks of ``w`` consecutive tasks and over the rows of ``Y``, with the s.e. over the block's values (§12.7 G10).

    Blocks never straddle a cut (end of burn-in, half boundary).

    Parameters
    ----------
    x : (T,) array
    Y : (n_rows, T) array
        One row per seed (NaN allowed).
    w : int
        Block width in tasks (≥ 1).
    cuts : sequence of int
        Task indices where a new block must start.

    Returns
    -------
    (xb, mean, se) arrays of length n_blocks
        ``xb`` the mean x of the block; ``se`` NaN for a block with one value.
    """
    x = np.asarray(x, float)
    Y = np.atleast_2d(np.asarray(Y, float))
    T = x.shape[0]
    bounds = sorted({0, T, *[int(c) for c in cuts if 0 < c < T]})
    xb, mb, sb = [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        for s in range(lo, hi, max(1, int(w))):
            e = min(s + max(1, int(w)), hi)
            vals = Y[:, s:e].ravel()
            vals = vals[np.isfinite(vals)]
            xb.append(x[s:e].mean())
            mb.append(vals.mean() if len(vals) else np.nan)
            sb.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan)
    return np.array(xb), np.array(mb), np.array(sb)


def overlay(ax, X, Y, color: str, cmap=None, median_color: str = "k"):
    """Thin per-row lines + median + 10–90 % band in one call (§12.3 B).

    Parameters
    ----------
    ax : matplotlib Axes
    X : (n,) x values
    Y : (m, n) one row per task (or seed)
    color : str
        Band colour (the readout's).
    cmap : str, optional
        Colormap for the row index (``viridis`` for task number); ``color`` otherwise.
    median_color : str
        Median line colour: black when the panel summarises one readout,
        the readout colour when several share a panel.

    Returns
    -------
    (n,) array
        The median.
    """
    Y = np.asarray(Y, float)
    cm = plt.get_cmap(cmap) if cmap else None
    for k, y in enumerate(Y):
        ax.plot(X, y, color=cm(k / max(len(Y) - 1, 1)) if cm else color, lw=LW_OVERLAY, alpha=ALPHA_TASK, zorder=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med, lo, hi = np.nanmedian(Y, 0), np.nanpercentile(Y, 10, 0), np.nanpercentile(Y, 90, 0)
    band(ax, X, lo, hi, color)
    ax.plot(X, med, color=median_color, lw=LW_MEAN, zorder=5)
    return med


def seeds_and_mean(ax, x, Y, ro: str) -> np.ndarray:
    """Seeds as thin lines (``alpha 0.5``) with the seed-mean on top, in the readout's colour.

    Returns
    -------
    (n,) array
        The seed mean (NaN-aware).
    """
    for y in Y:
        plot_readout(ax, x, y, ro, lw=LW_OVERLAY, alpha=ALPHA_SEED, zorder=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmean(np.asarray(Y, float), 0)
    plot_readout(ax, x, m, ro, lw=LW_MEAN, zorder=5)
    return m


def dumbbell(ax, x, y0: float, y1: float, se: float, color: str, flagged: bool = False, hollow: bool = False) -> None:
    """Open circle at ``y0``, filled circle at ``y1``, joined, ±3 s.e. error bars; ``flag_mark`` when flagged.

    Parameters
    ----------
    ax : matplotlib Axes
    x : float
        Categorical position.
    y0, y1, se : float
    color : str
    flagged : bool
    hollow : bool
        Hollow half-1 marker (ungated readouts).
    """
    ax.errorbar([x, x], [y0, y1], yerr=3 * se, fmt="none", ecolor=GREY_REF, elinewidth=LW_REF, zorder=2)
    ax.plot([x, x], [y0, y1], color=color, lw=LW_TRACE, zorder=3)
    ax.plot(x, y0, "o", mfc="white", mec=color, ms=4.5, zorder=4)
    ax.plot(x, y1, "o", mfc="white" if hollow else color, mec=color, ms=4.5, zorder=4)
    if flagged:
        flag_mark(ax, x, y1)


def cell_ticks(ax, cells, labels: bool = True) -> None:
    """Nested horizontal labels for a categorical (κ, δ₀, λ) cell axis (§12.6 C2-B).

    Every tick shows λ; the first cell of each δ₀ group adds ``δ₀ = …`` on
    a second line and the first cell of each κ group adds ``κ = …`` on a
    third; κ groups are separated by a thin grey rule.

    Parameters
    ----------
    ax : matplotlib Axes
    cells : list of (kappa_label, delta0, lam), in plotting order
    labels : bool
        Write the labels (bottom row only).
    """
    n = len(cells)
    ax.set_xticks(range(n))
    multi_k = len({c[0] for c in cells}) > 1
    labs = []
    for i, (k, d, l) in enumerate(cells):
        s = fmt(l)
        if i == 0 or cells[i - 1][:2] != (k, d):
            s += "\n" + rf"$\delta_0$={fmt(d)}"
        if multi_k and (i == 0 or cells[i - 1][0] != k):
            s += "\n" + rf"$\kappa$={'√N' if k == 'sqrtN' else fmt(k)}"
        if i > 0 and cells[i - 1][0] != k:
            ax.axvline(i - 0.5, color=GREY_REF, lw=LW_REF, zorder=1)
        labs.append(s)
    ax.set_xticklabels(labs if labels else [], fontsize=FS_MIN)
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(False, axis="x")
    if labels:
        ax.set_xlabel(r"$\lambda$ (per cell)")


def dist_by_readout(ax, values_by_readout: dict, band: float | None = 3.0, clip: float | None = None,
                    na: dict | None = None, labels: bool = True, seed: int = 0) -> None:
    """The per-cell values of each readout as points in a slot, a box behind them only when n ≥ ``MIN_BOX_N`` (§12.7 G5).

    Parameters
    ----------
    ax : matplotlib Axes
    values_by_readout : dict
        ``{readout: 1-D array}``; NaNs dropped.  Every readout in ``READOUT_COLOR``
        that appears here or in ``na`` gets a slot, in the fixed order.
    band : float, optional
        Draws ``zero_band(−band, band)``; ``None`` draws only the zero line.
    clip : float, optional
        Values beyond ±clip are plotted at the edge by ``edge_markers`` (call after limits are set:
        this function records them on ``ax._mup_clipped``).
    na : dict, optional
        ``{readout: reason}`` slots that are n/a by construction (hatched, reason printed).
    labels : bool
        Readout names under the slots (bottom row only).
    seed : int
        Jitter seed (deterministic).
    """
    from mup.config import is_gated
    na = na or {}
    ros = [r for r in READOUT_COLOR if r in values_by_readout or r in na]
    rng = np.random.default_rng(seed)
    if band is not None:
        zero_band(ax, -band, band)
    else:
        ref_line(ax, 0.0)
    clipped_x, clipped_y = [], []
    for i, ro in enumerate(ros):
        if ro in na:
            na_slot(ax, na[ro], x=i)
            continue
        v = np.asarray(values_by_readout[ro], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        col = READOUT_COLOR[ro]
        if len(v) >= MIN_BOX_N:
            bp = ax.boxplot([v], positions=[i], widths=0.6, whis=(5, 95), patch_artist=True, manage_ticks=False,
                            showfliers=False, zorder=2,
                            boxprops=dict(facecolor=col, edgecolor=col, alpha=0.25, lw=LW_REF),
                            whiskerprops=dict(color=col, lw=LW_REF), capprops=dict(color=col, lw=LW_REF),
                            medianprops=dict(lw=0))
            if not is_gated(ro):
                for b in bp["boxes"]:
                    b.set_hatch("///"); b.set_facecolor("white"); b.set_alpha(1.0)
        vin = np.clip(v, -clip, clip) if clip is not None else v
        xj = i + rng.uniform(-0.22, 0.22, size=len(v))
        ax.plot(xj, vin, "o", ms=4, mfc=col if is_gated(ro) else "white", mec=col, mew=0.9, ls="none", zorder=4, alpha=0.9)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="k", lw=LW_TRACE, zorder=5)
        if clip is not None:
            out = np.abs(v) > clip
            clipped_x += list(xj[out]); clipped_y += list(v[out])
    ax._mup_clipped = (np.array(clipped_x), np.array(clipped_y))
    ax._mup_no_const_check = True
    ax.set_xticks(range(len(ros)))
    ax.set_xticklabels([READOUT_LABEL[r] for r in ros] if labels else [], rotation=90, fontsize=FS_MIN)
    ax.set_xlim(-0.7, len(ros) - 0.3)
    ax.grid(False, axis="x")


def flag_table(ax, counts: dict, quantities: list, readouts: list, na: dict | None = None, row_labels: bool = True) -> None:
    """Table of flagged-cell counts ``k/n`` per (quantity row, readout column) on a white → red fill (§12.6 C2-E).

    Parameters
    ----------
    ax : matplotlib Axes
    counts : dict
        ``{(quantity, readout): (k, n)}``.
    quantities : list of str
        Row keys (top to bottom), labelled through ``QUANTITY_LABEL``.
    readouts : list of str
        Column keys in the fixed order; headers in the readout colour.
    na : dict, optional
        ``{(quantity, readout): reason}`` cells shown hatched with "—".
    row_labels : bool
        Write the quantity labels (left column only).
    """
    na = na or {}
    nq, nr = len(quantities), len(readouts)
    frac = np.full((nq, nr), np.nan)
    for i, q in enumerate(quantities):
        for j, ro in enumerate(readouts):
            if (q, ro) in counts and counts[(q, ro)][1] > 0:
                k, n = counts[(q, ro)]
                frac[i, j] = k / n
    ax.imshow(np.nan_to_num(frac, nan=0.0), cmap=FLAG_CMAP, norm=PowerNorm(0.5, vmin=0, vmax=1), aspect="auto", zorder=1)
    for i, q in enumerate(quantities):
        for j, ro in enumerate(readouts):
            if (q, ro) in na or (q, ro) not in counts:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, hatch="///", fc="white", ec=GREY_NA, lw=0, zorder=2))
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=GREY_CAPTION, zorder=3)
            else:
                k, n = counts[(q, ro)]
                ax.text(j, i, f"{k}/{n}", ha="center", va="center", fontsize=9, color="k" if frac[i, j] < 0.6 else "white", zorder=3)
    for i in range(nq + 1):
        ax.axhline(i - 0.5, color="white", lw=1.0, zorder=2.5)
    for j in range(nr + 1):
        ax.axvline(j - 0.5, color="white", lw=1.0, zorder=2.5)
    ax.set_xticks(range(nr)); ax.set_xticklabels([READOUT_LABEL[r] for r in readouts], rotation=90, fontsize=FS_MIN)
    for t, ro in zip(ax.get_xticklabels(), readouts):
        t.set_color(READOUT_COLOR[ro])
    ax.set_yticks(range(nq)); ax.set_yticklabels([qlabel(q) for q in quantities] if row_labels else [], fontsize=FS_MIN)
    ax.set_xlim(-0.5, nr - 0.5); ax.set_ylim(nq - 0.5, -0.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax._mup_no_const_check = True


def mhalf_ticks(ax, m_by_readout: dict) -> None:
    """m½ of half 0 (open) and half 1 (filled) per readout in a strip at the bottom of the panel (§12.7.2).

    Drawn in axes-fraction y so the ticks never move the y-limits.

    Parameters
    ----------
    ax : matplotlib Axes
    m_by_readout : dict
        ``{readout: (m_half_0, m_half_1)}``; NaN entries are skipped.
    """
    tr = ax.get_xaxis_transform()
    for i, ro in enumerate(r for r in READOUT_COLOR if r in m_by_readout):
        m0, m1 = m_by_readout[ro]
        y = 0.04 + 0.05 * i
        col = READOUT_COLOR[ro]
        if np.isfinite(m0) and np.isfinite(m1):
            ax.plot([m0, m1], [y, y], color=col, lw=LW_REF, transform=tr, zorder=6, scaley=False, scalex=False)
        if np.isfinite(m0):
            ax.plot([m0], [y], "o", ms=4, mfc="white", mec=col, transform=tr, zorder=6, scaley=False, scalex=False)
        if np.isfinite(m1):
            ax.plot([m1], [y], "o", ms=4, mfc=col, mec=col, transform=tr, zorder=6, scaley=False, scalex=False)


def drift_map(ax, z, beta, d0s, lams, beta_fmt=None, zmax: float = 5.0):
    """``imshow`` of the significance β/se on the (δ₀ rows, λ columns) grid with β printed in each cell (§12.6 C2-C).

    Parameters
    ----------
    ax : matplotlib Axes
    z : (n_d0, n_lam) array
        β/se(β); NaN for missing cells (hatched).
    beta : (n_d0, n_lam) array
        β, printed to 2 significant figures.
    d0s, lams : sequences for the tick labels (through ``fmt``)
    beta_fmt : callable, optional
        Text of a β value (default ``fmt2``).
    zmax : float
        Fixed colour limit (±zmax, the same in every figure).

    Returns
    -------
    matplotlib AxesImage
    """
    z, beta = np.asarray(z, float), np.asarray(beta, float)
    beta_fmt = beta_fmt or fmt2
    im = ax.imshow(np.nan_to_num(z, nan=0.0), cmap="RdBu_r", vmin=-zmax, vmax=zmax, origin="lower", aspect="auto", zorder=1)
    for (a, b), v in np.ndenumerate(z):
        if not np.isfinite(v):
            ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, hatch="///", fc="white", ec=GREY_NA, lw=0, zorder=2))
            continue
        if abs(v) > 3:
            ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="k", lw=LW_TRACE, zorder=3))
        ax.text(b, a, beta_fmt(beta[a, b]), ha="center", va="center", fontsize=FS_MIN,
                color="white" if abs(v) > 0.6 * zmax else "k", zorder=4)
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([fmt(l) for l in lams])
    ax.set_yticks(range(len(d0s))); ax.set_yticklabels([fmt(d) for d in d0s])
    ax.grid(False)
    ax._mup_no_const_check = True
    return im
