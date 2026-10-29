"""Figure standard (PLAN.md §12.0): the only style file.

Inspection figures: PNG only, sized by panel (2.6 in × 2.0 in) so every
panel is legible at 100 % zoom, one style for every figure of
``analyze.py``, ``plot_norms.py``, ``plot_check2.py`` and ``validate.py``.
No script sets ``rcParams`` or picks a colour itself; everything comes
from here.
"""
from __future__ import annotations

import logging
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

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
#: sign colours (half-window ratios): up / down
UP, DOWN = "#D55E00", "#0072B2"
GREY_REF, GREY_BURNIN, GREY_CAPTION, RED_FLAG = "0.4", "0.92", "0.3", "#CC0000"
#: line widths of PLAN.md §12.0 (layering bottom → top)
LW_REF, LW_OVERLAY, LW_TRACE, LW_MEAN, LW_EVENT = 0.8, 0.6, 1.6, 2.2, 0.8
ALPHA_TASK, ALPHA_SEED, ALPHA_BAND = 0.08, 0.5, 0.2
PANEL = (2.6, 2.0)          # inches, w × h
TITLE_IN = 0.55             # inches reserved at the top for suptitle + caption
MAX_COLS = 6
FS_MIN = 8


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
    })


apply_style()


# ---------------------------------------------------------------------------
# figure construction and saving
# ---------------------------------------------------------------------------
def grid(nrows: int, ncols: int, sharex="col", sharey="row", panel=PANEL, extra_w: float = 0.0, **kw):
    """``plt.subplots`` at a fixed physical size per panel, constrained layout, ≤ 4 ticks per linear axis.

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
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols + 1.4 + extra_w, h * nrows + 1.3),
                             sharex=sharex, sharey=sharey, squeeze=False, layout="constrained", **kw)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - TITLE_IN / (h * nrows + 1.3)))   # keep the title band free
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


def _check(fig) -> None:
    """Figure-level checks of PLAN.md §12.0; each failure is a warning."""
    if fig._suptitle is None:
        warnings.warn("figure without a suptitle")
    cols = {round(ax.get_position().x0, 3) for ax in fig.axes if ax.get_subplotspec() is not None}
    if len(cols) > MAX_COLS:
        warnings.warn(f"figure is {len(cols)} panels wide (> {MAX_COLS})")
    w_in = fig.get_size_inches()[0]
    for ax in fig.axes:
        if ax.get_subplotspec() is None:
            continue
        if ax.get_position().width * w_in < 2.0 - 1e-6:
            warnings.warn("panel narrower than 2 in")
            break
    for ax in fig.axes:
        for a, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            if isinstance(a.get_major_locator(), FixedLocator):      # categorical axis: one tick per category
                continue
            ticks = [t for t in a.get_majorticklocs() if min(lim) <= t <= max(lim)]
            if len(ticks) > 6:
                warnings.warn(f"axis with {len(ticks)} ticks (> 6)")
                break
    for t in fig.findobj(Text):
        if t.get_text().strip() and t.get_fontsize() < FS_MIN:
            warnings.warn(f"text smaller than {FS_MIN} pt: {t.get_text()[:30]!r}")
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
            if a.get_scale() == "log" and not isinstance(a.get_major_locator(), LogLocator):
                logaxis(ax, axis_name)
            elif a.get_scale() == "log":
                a.set_major_locator(LogLocator(base=10, numticks=4)); a.set_minor_formatter(NullFormatter())
    fig.canvas.draw()          # final tick positions for the checks
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
# titles, names, legends
# ---------------------------------------------------------------------------
_GREEK = {"κ": r"$\kappa$", "δ₀": r"$\delta_0$", "λ": r"$\lambda$", "η": r"$\eta$"}


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
        Fields fixed in the figure; ``None`` is omitted.
    extra : str
        Appended after the cell (e.g. a readout name).
    """
    parts = []
    if kappa is not None:
        parts.append(rf"$\kappa$ = {kappa}")
    if d0 is not None:
        parts.append(rf"$\delta_0$ = {d0}")
    if lam is not None:
        parts.append(rf"$\lambda$ = {lam}")
    if solver is not None:
        parts.append({"pop": "population", "sgd": "SGD"}.get(solver, solver) + (rf", $\eta$ = {eta}" if eta is not None else ""))
    if extra:
        parts.append(mathify(extra))
    h = fig.get_size_inches()[1]
    fig.suptitle(", ".join(parts) if parts else mathify(extra) or " ", y=1 - 0.06 / h, va="top")


def caption(fig, text: str) -> None:
    """One-line caption under the suptitle (9 pt, grey) saying what overlays and bands are.

    Parameters
    ----------
    fig : matplotlib Figure
    text : str
    """
    h = fig.get_size_inches()[1]
    fig.text(0.5, 1 - 0.30 / h, mathify(text), ha="center", va="top", fontsize=9, color=GREY_CAPTION)


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


def readout_legend(ax, readouts=None, loc="upper right", **kw):
    """Legend of readout colours/styles in the fixed order of PLAN.md §4.

    Parameters
    ----------
    ax : matplotlib Axes
    readouts : iterable of str, optional
        Subset to list (default: all five).
    loc : str
    **kw
        Passed to ``ax.legend``.

    Returns
    -------
    matplotlib Legend
    """
    ros = [r for r in READOUT_COLOR if readouts is None or r in set(readouts)]
    handles = [Line2D([], [], color=READOUT_COLOR[r], ls=READOUT_STYLE[r], lw=LW_TRACE, label=READOUT_LABEL[r]) for r in ros]
    return ax.legend(handles=handles, loc=loc, **kw)


def label_cols_N(axes, Ns) -> None:
    """Title the top row of a panel grid ``N = …``.

    Parameters
    ----------
    axes : 2-D array of Axes
    Ns : sequence of int
    """
    for ax, N in zip(np.atleast_2d(axes)[0], Ns):
        ax.set_title(f"N = {N}")


def n_shade(i: int, n: int) -> str:
    """Colour for the i-th of n widths (light for small N, dark for large)."""
    return N_SHADES[int(round(i * (len(N_SHADES) - 1) / max(n - 1, 1)))]


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


def band(ax, x, lo, hi, color: str, alpha: float = ALPHA_BAND) -> None:
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
        ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=alpha, lw=0, zorder=3)


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


def dumbbell(ax, x, y0: float, y1: float, se: float, color: str, flagged: bool = False) -> None:
    """Open circle at ``y0``, filled circle at ``y1``, joined, ±3 s.e. error bars; black + red × when flagged.

    Parameters
    ----------
    ax : matplotlib Axes
    x : float
        Categorical position.
    y0, y1, se : float
    color : str
    flagged : bool
    """
    col = "k" if flagged else color
    ax.errorbar([x, x], [y0, y1], yerr=3 * se, fmt="none", ecolor=GREY_REF, elinewidth=LW_REF, zorder=2)
    ax.plot([x, x], [y0, y1], color=col, lw=LW_TRACE, zorder=3)
    ax.plot(x, y0, "o", mfc="white", mec=col, ms=4.5, zorder=4)
    ax.plot(x, y1, "o", color=col, ms=4.5, zorder=4)
    if flagged:
        ax.plot(x, max(y0, y1), "x", color=RED_FLAG, ms=5, mew=1.2, zorder=5)


def drift_map(ax, beta, d0s, lams, vmax: float, se=None):
    """``imshow`` of β on the (δ₀ rows, λ columns) grid, ``RdBu_r`` centred at zero (§12.6 C2-C).

    Parameters
    ----------
    ax : matplotlib Axes
    beta : (n_d0, n_lam) array, NaN for missing cells (hatched)
    d0s, lams : sequences for the tick labels
    vmax : float
        Shared colour limit.
    se : (n_d0, n_lam) array, optional
        Cells with |β| > 3·se are outlined in black.

    Returns
    -------
    matplotlib AxesImage
    """
    beta = np.asarray(beta, float)
    im = ax.imshow(beta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    for (a, b), v in np.ndenumerate(beta):
        if not np.isfinite(v):
            ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, hatch="///", fill=False, ec="0.6", lw=0))
        elif se is not None and abs(v) > 3 * se[a, b]:
            ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="k", lw=LW_TRACE))
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([str(l) for l in lams])
    ax.set_yticks(range(len(d0s))); ax.set_yticklabels([str(d) for d in d0s])
    ax.grid(False)
    return im


def zero_band(ax, lo: float, hi: float, axis: str = "y") -> None:
    """Grey band (``GREY_BURNIN``) between ``lo`` and ``hi`` plus a grey zero line (the flag threshold of §12.6 C2-E).

    Parameters
    ----------
    ax : matplotlib Axes
    lo, hi : float
    axis : {"y", "x"}
    """
    (ax.axhspan if axis == "y" else ax.axvspan)(lo, hi, color=GREY_BURNIN, lw=0, zorder=0)
    ref_line(ax, 0.0, axis)


def boxes_by_readout(ax, data: dict, labels: bool = True) -> None:
    """One box plot per readout in the fixed order and colour (§12.6 C2-E).

    Box = IQR, whiskers = 5–95 %, median line black, outliers as small dots
    in the readout colour; ungated readouts hatched (dashing does not apply
    to a box).  NaNs are dropped; a readout with no finite value gets no box.

    Parameters
    ----------
    ax : matplotlib Axes
    data : dict
        ``{readout: 1-D array}`` of the per-cell statistic.
    labels : bool
        Write the readout labels under the boxes (bottom row only).
    """
    from mup.config import is_gated
    ros = [r for r in READOUT_COLOR if r in data]
    for i, ro in enumerate(ros):
        v = np.asarray(data[ro], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        col = READOUT_COLOR[ro]
        bp = ax.boxplot([v], positions=[i], widths=0.6, whis=(5, 95), patch_artist=True, manage_ticks=False,
                        boxprops=dict(facecolor=col, edgecolor=col, alpha=0.5, lw=LW_REF),
                        whiskerprops=dict(color=col, lw=LW_REF), capprops=dict(color=col, lw=LW_REF),
                        medianprops=dict(color="k", lw=LW_TRACE),
                        flierprops=dict(marker="o", ms=2.5, mfc=col, mec="none", alpha=0.7))
        if not is_gated(ro):
            for b in bp["boxes"]:
                b.set_hatch("///"); b.set_facecolor("white"); b.set_alpha(1.0)
    ax.set_xticks(range(len(ros)))
    ax.set_xticklabels([READOUT_LABEL[r] for r in ros] if labels else [], rotation=90, fontsize=FS_MIN)
    ax.set_xlim(-0.7, len(ros) - 0.3)
    ax.grid(False, axis="x")


def flag_bars(ax, fractions: dict, counts: dict, quantities: list, chance: float = 0.0027, labels: bool = True) -> None:
    """Horizontal grouped bars of the flagged-cell fraction, y = quantity, one bar per readout (§12.6 C2-E).

    Parameters
    ----------
    ax : matplotlib Axes
    fractions, counts : dict
        ``{(quantity, readout): fraction}`` and ``{(quantity, readout): "k/n"}``.
    quantities : list of str
        Row order (top to bottom).
    chance : float
        Fraction expected by chance at the 3-s.e. level, drawn as a grey line.
    labels : bool
        Write the quantity names on the y-axis (left column only).
    """
    from mup.config import is_gated
    ros = [r for r in READOUT_COLOR if any(k[1] == r for k in fractions)]
    nq, nr = len(quantities), max(len(ros), 1)
    hgt = 0.8 / nr
    for qi, q in enumerate(quantities):
        for ri, ro in enumerate(ros):
            if (q, ro) not in fractions:
                continue
            y = nq - 1 - qi + (ri - (nr - 1) / 2) * hgt
            f = fractions[(q, ro)]
            ax.barh(y, f, height=hgt * 0.9, color=READOUT_COLOR[ro], hatch=None if is_gated(ro) else "///",
                    edgecolor=READOUT_COLOR[ro], lw=0, zorder=3)
            ax.text(max(f, 0.0) + 0.01, y, counts[(q, ro)], va="center", ha="left", fontsize=FS_MIN, color=GREY_CAPTION)
    ref_line(ax, chance, "x")
    ax.set_yticks(range(nq))
    ax.set_yticklabels(list(reversed(quantities)) if labels else [], fontsize=FS_MIN)
    ax.set_ylim(-0.6, nq - 0.4)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.grid(False, axis="y")
