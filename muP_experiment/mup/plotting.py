"""Shared figure conventions: one fixed colour per readout, thin lines, single axes."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe–Ito hues, bound to the entity (readout), never to rank or order.
READOUT_COLOR = {"learned": "#0072B2", "frozen": "#D55E00", "heads": "#009E73"}
READOUT_LABEL = {"learned": "learned shared", "frozen": "frozen shared", "heads": "heads"}
# sequential shades for width N (light → dark), one hue
N_SHADES = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "legend.frameon": False, "axes.titlesize": 9,
})


def n_shade(i, n):
    idx = int(round(i * (len(N_SHADES) - 1) / max(n - 1, 1)))
    return N_SHADES[idx]
