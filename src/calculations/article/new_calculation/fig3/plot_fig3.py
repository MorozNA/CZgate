import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
})

path = 'data/data_a/'
fidelity1 = np.loadtxt(path + 'fidelity_1.txt')
fidelity3 = np.loadtxt(path + 'fidelity_3.txt')
fidelity5 = np.loadtxt(path + 'fidelity_5.txt')

def per_gate_fidelity(F):
    return np.insert(F[1:] / F[:-1], 0, F[0])

f1 = per_gate_fidelity(fidelity1)
f3 = per_gate_fidelity(fidelity3)
f5 = per_gate_fidelity(fidelity5)

N = np.arange(1, len(f1) + 1)

fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)

colors_1 = ['blue', 'green', 'red']
colors_2 = ['#1f77b4', '#2ca02c', '#d62728']


styles = [
    (f1, colors_2[0],  'o', r"$T = 1~\mu\mathrm{K}$"),  # circle
    (f3, colors_2[1], '^', r"$T = 3~\mu\mathrm{K}$"),  # triangle
    (f5, colors_2[2],   's', r"$T = 5~\mu\mathrm{K}$"),  # square
]

# prettier data points only
for f, color, marker, label in styles:
    ax.plot(
        N, f,
        linestyle='None',
        marker=marker,
        ms=5.0,               # marker size
        mew=0.8,              # marker edge width
        mfc='white',          # marker face color
        mec=color,            # marker edge color
        alpha=0.9,
        label=label
    )




# straight-line fits (same N range as shown)
points_to_fit = 5
N_fit = np.linspace(1, points_to_fit, 100)  # or up to len(f1)
for f, color, _, _ in styles:
    a, b = np.polyfit(N[:points_to_fit], f[:points_to_fit], 1)  # fit first 10 points
    ax.plot(N_fit, a*N_fit + b,
            color=color, lw=1.0, ls='--', zorder=10)


# horizontal lines from each starting point (N=1 to last shown N)
x0, x1 = 1, points_to_fit
for f, color, _, _ in styles:
    y0 = f[0]          # starting value at N=1
    ax.hlines(y0, x0, x1,
              colors=color,
              linestyles=':',
              linewidth=0.85,
              alpha=0.85,
              zorder=0)


# axes: show only N = 1..10
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$f_N = F_N / F_{N-1}$')

# ymin = min(f1.min(), f3.min(), f5.min())
# ymax = max(f1.max(), f3.max(), f5.max())
# margin = 0.25 * (ymax - ymin)
# ax.set_ylim(ymin - margin, ymax + margin)

ax.set_xlim(1 - 0.1, points_to_fit + 0.1)
ax.set_xticks(np.arange(1, points_to_fit+1, 1))
ymin = 0.9985
ymax = 1.0
eps = 0.0001
ax.set_ylim(ymin -  eps, ymax + eps)
ax.set_yticks(np.linspace(ymin, ymax, 4))

ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.8)
ax.tick_params(direction="in", which="both", top=True, right=True)
ax.legend(
    loc="lower left",
    frameon=True,
    bbox_to_anchor=(0.07, 0.07)  # (x, y) in axes coords
)

fig.savefig(path + "figure2_fN_points_fit.png",
            bbox_inches="tight", facecolor="white")
fig.savefig(path + "figure2_fN_points_fit.pdf",
            bbox_inches="tight", facecolor="white")

plt.show()