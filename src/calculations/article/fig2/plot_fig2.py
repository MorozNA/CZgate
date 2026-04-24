import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Global matplotlib settings (APS style)
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
})

# -----------------------------
# Helper function to load panel data
# -----------------------------
def load_data(effect_name):
    path = "data/data_" + effect_name + "/"

    fidelity_alg = np.loadtxt(path + "fidelity_alg_pulse_1.txt")
    fidelity_no_motion = np.loadtxt(path + "fidelity_alg_pulse_0.txt")
    fidelity_full = np.loadtxt(path + "fidelity_full.txt")
    fidelity_alg_decomp = np.loadtxt(path + "fidelity_hybrid_pulse_1.txt")

    fidelity_naive = fidelity_full[0]

    iterations = len(fidelity_alg)
    x = np.arange(1, iterations + 1)

    return x, fidelity_alg, fidelity_no_motion, fidelity_full, fidelity_alg_decomp, fidelity_naive



# Names just to keep the order consistent
curve_labels = [
    r"$\rho_{\mathrm{full}}(t_N)$", r"$\rho(t_N)$", r"$\rho_{\mathrm{fac}}(t_N)$", r"$\rho_{in}(t_N)$", r"$1 - F^N$"]

# Example: style lists for panel (a)
# colors_a = ["tab:red", "#08306b", "#1f5f9d", "#6baed6", "0.45"]
# colors_a = ["magenta", "#08306b", "#1f5f9d", "#6baed6", "0.45"]
# colors_a = ["magenta", "#08306b", "#2171b5", "#6baed6", "#9ecae1"]
colors_a = ["magenta", "#08306b", "#174f7e", "#5287a6", "#7b9daf"]
ls_a     = ["dashed", "solid", "dashed", "dashdot", "dotted"]
# lws_a    = [1.15, 1.15, 1.15, 1.15, 1.15]
lws_a    = [0.9, 0.9, 0.9, 0.9, 0.9]
alph_a   = [0.7, 0.95, 0.9, 1.0, 1.0]

colors_b = colors_a
ls_b     = ls_a
lws_b    = [0.85, 1.2, 0.9, 0.9, 0.9]
alph_b   = [0.8, 1.0, 1.0, 1.0, 1.0]

zord = [10, 8, 6, 4, 2]

# colors_a = ["#08306b", "#4292c6", "#6baed6", "grey", "#c72b29"]
# lws_a    = [1.1, 1.2, 1.1, 1.1, 1.1]
# ls_a     = ["solid", "dashed", "dashdot", "dotted", "dotted"]
# alph_a   = [0.9, 1.0, 0.95, 0.85, 0.8]
#
# # Example: style lists for panel (b) (can be different)
# colors_b = ["#08306b", "#4292c6", "#6baed6", "grey", "#c72b29"]
# lws_b    = [1.3, 1.3, 1.0, 1.0, 1.0]
# ls_b     = ["solid", "dashed", "dashdot", "dotted", "dotted"]
# alph_b   = [0.9, 0.9, 0.9, 0.8, 0.7]

def plot_panel(ax, effect_name, panel_label,
               colors, lws, ls, alph):
    x, fidelity_alg, fidelity_no_motion, fidelity_full, \
        fidelity_alg_decomp, fidelity_naive = load_data(effect_name)

    curves_y = [
        1 - fidelity_full,
        1 - fidelity_alg,
        1 - fidelity_alg_decomp,
        1 - fidelity_no_motion,
        1 - fidelity_naive**x,
    ]

    for i, y in enumerate(curves_y):
        ax.plot(
            x, y,
            color=colors[i],
            linewidth=lws[i],
            linestyle=ls[i],
            label=curve_labels[i],
            alpha=alph[i],
            zorder=zord[i],  # keep special zorder if you want
        )

    ax.set_xlim([-0.75, 30.75])
    ax.set_ylim([-0.001, 0.044])
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04])
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

    ax.text(0.06, 0.96, panel_label,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top")

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.4), sharey=True)
ax_a, ax_b = axes

plot_panel(ax_a, "a", "(a)", colors_a, lws_a, ls_a, alph_a)
plot_panel(ax_b, "b", "(b)", colors_b, lws_b, ls_b, alph_b)

axes[0].set_ylabel(r"$1 - F_N$", labelpad=6)
for ax in axes:
    ax.set_xlabel(r"$N$", labelpad=4)

handles, labels = ax_a.get_legend_handles_labels()

import copy
handles[0] = copy.copy(handles[0])
handles[0].set_linestyle('-')
handles[0].set_dashes([3.3, 1.4, 3.3, 1.4])   # dash, gap, dot, gap
handles[2] = copy.copy(handles[3])
handles[2].set_linestyle('-')
handles[2].set_dashes([3.3, 1.4, 3.3, 1.4])   # dash, gap, dot, gap
handles[3] = copy.copy(handles[3])
handles[3].set_linestyle('-.')
handles[3].set_dashes([4.15, 1.4, 1.4, 1.4])   # dash, gap, dot, gap
# handles[3].set_dash_capstyle('round')   # makes the dot look like a dot

leg = ax_b.legend(handles, labels,
                  loc="upper right",
                  ncol=1,
                  frameon=False,
                  fontsize=8.5,
                  handlelength=1.3,
                  columnspacing=1.0,
                  borderpad=0.3,
                  bbox_to_anchor=(0.75, 0.88))
leg.get_frame().set_alpha(0.9)


plt.savefig("data/full_figure_ab_clrs.pdf", bbox_inches="tight")
plt.savefig("data/full_figure_ab_clrs.png", dpi=300, bbox_inches="tight")
plt.show()