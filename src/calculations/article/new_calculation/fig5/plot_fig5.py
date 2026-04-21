import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter

from src.y_operator_deltaR.params import HBAR, OM_small, kB
from src.algorithm.other_tools import get_rho_T0

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
plt.style.use("tableau-colorblind10")

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
T_muK = 1
n = 250
path = f"data/T{T_muK}/n{n}/"

dms = np.loadtxt(path + f"dm_diags_T{T_muK}.txt")
n_averages = np.loadtxt(path + f"n_avg_T{int(T_muK)}.txt")
temps_n_avg = HBAR * OM_small * (n_averages + 0.5) / kB

iter_nums = [1, 50, 100, 200]
energy_levels = np.arange(n)

colors = ["#CCCCCC", "#969696", "#4D4D4D", "#E41A1C"]
iter_hist = 200
n_max_log = 81   # from PLOT1
cutoff = 32

# ------------------------------------------------------------
# Main figure: HISTOGRAM (from PLOT1)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.2, 2.4), constrained_layout=True)


ev = energy_levels[:cutoff]
bar_width = 0.85

# Initial thermal at T0
rho_th0 = get_rho_T0(T_muK * 1e-6, n)
y0 = np.diag(rho_th0)[energy_levels[:cutoff]]

# Numerical at iter_hist
y_num = dms[iter_hist - 1][energy_levels[:cutoff]]

# Thermal fit at T_iter_hist
T_fit_hist = temps_n_avg[iter_hist - 1]
rho_th_hist = get_rho_T0(T_fit_hist, n)
y_fit_hist = np.diag(rho_th_hist)[energy_levels[:cutoff]]

# T0: gray background bars
ax.bar(ev, y0, width=bar_width, color="#CCCCCC", alpha=0.8, zorder=1)
ax.plot(
    ev, y0,
    color="#CCCCCC", lw=1.0, ls="-", zorder=1.1, alpha=1.0
)

# Numerical rho(t_iter_hist): red histogram
ax.bar(ev, y_num, width=bar_width, color="#E41A1C", alpha=0.5, zorder=3.0)
ax.plot(
    ev, y_num,
    color="#E41A1C", lw=1.0, ls="-", zorder=1.3, alpha=0.9
)

# Thermal fit: dashed line through bin centers
ax.plot(
    ev, y_fit_hist,
    color="#E41A1C", lw=1.0, ls="--", zorder=1.4, alpha=0.9
)

ax.set_xlabel(r"$v_z$")
ax.set_ylabel("$P$ $(v_z)$")
ax.set_xlim(-1, cutoff-1)
ax.set_ylim(0, None)
ax.tick_params(direction="in", which="both", top=True, right=True, length=6)
ax.tick_params(which="minor", length=3)
ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.45)

# Optional: histogram legend (same as PLOT1, uncomment if you want it)
# hist_handles = [
#     Patch(facecolor="#CCCCCC", edgecolor="none", alpha=0.9),
#     Patch(facecolor="#E41A1C", edgecolor="none", alpha=0.35),
#     Line2D([], [], color="#E41A1C", lw=1.2, ls="--"),
# ]
# hist_labels = [
#     rf"$T_0 = {T_muK:.1f}\,\mu\mathrm{{K}}$",
#     rf"$\rho(t_{{{iter_hist}}})$",
#     rf"Thermal fit, $T_{{{iter_hist}}}={temps_n_avg[iter_hist-1]/1e-6:.1f}\,\mu\mathrm{{K}}$",
# ]
# ax.legend(
#     hist_handles, hist_labels,
#     loc="upper right",
#     frameon=False,
#     fontsize=7,
#     handlelength=1.4,
#     handletextpad=0.5,
#     borderaxespad=0.2,
#     labelspacing=0.3,
# )

# ------------------------------------------------------------
# Inset: LOG-SCALE semilogy (top plot from PLOT1)
# ------------------------------------------------------------
ax_ins = ax.inset_axes([0.29, 0.32, 0.65, 0.6])  # position similar to PLOT2

for i, it in enumerate(iter_nums):
    T_fit = temps_n_avg[it - 1]
    rho_th = get_rho_T0(T_fit, n)

    y_fit = np.diag(rho_th)[energy_levels]
    y_data = dms[it - 1][energy_levels]
    color = colors[i]

    m_data = y_data > 0
    m_fit = y_fit > 0

    ax_ins.semilogy(
        energy_levels[m_data], y_data[m_data],
        color=color, lw=0.9, ls="-"
    )
    ax_ins.semilogy(
        energy_levels[m_fit], y_fit[m_fit],
        color=color, lw=0.9, ls="--"
    )

ax_ins.set_xlim(0, n_max_log)
ax_ins.set_ylim(1e-4, 1.0e-1)
ax_ins.set_xlabel(r"$v_z$", fontsize=6, labelpad=1)
ax_ins.set_ylabel("$P$ $(v_z)$", fontsize=6, labelpad=2)
ax_ins.tick_params(direction="in", which="both", labelsize=5, length=3, pad=2)
ax_ins.tick_params(which="minor", length=1.5)
ax_ins.yaxis.set_major_locator(LogLocator(base=10))
ax_ins.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax_ins.yaxis.set_minor_formatter(NullFormatter())
ax_ins.grid(True, which="major", linestyle=":", linewidth=0.4, alpha=0.4)

for spine in ax_ins.spines.values():
    spine.set_linewidth(0.6)

# Legend on inset (same as top plot of PLOT1)
temp_handles = [
    Line2D([], [], color=colors[i], lw=1.0)
    for i in range(len(iter_nums))
]
temp_labels = [
    rf"$T_{{{it}}}={temps_n_avg[it-1]/1e-6:.1f}\,\mu\mathrm{{K}}$"
    for it in iter_nums
]

leg1 = ax_ins.legend(
    temp_handles, temp_labels,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.97),
    frameon=False,
    ncol=1,
    handlelength=1.6,
    columnspacing=0.6,
    handletextpad=0.4,
    borderaxespad=0.2,
    fontsize=6,
)
ax_ins.add_artist(leg1)


# after you have ax and ax_ins
# ax.set_axisbelow(False)
# ax_ins.set_axisbelow(False)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
fig.savefig(path + f"distribution_T{T_muK}_hist_with_inset.pdf", bbox_inches="tight")
fig.savefig(path + f"distribution_T{T_muK}_hist_with_inset.png", bbox_inches="tight")
plt.show()