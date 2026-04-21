import numpy as np
import matplotlib.pyplot as plt

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

ind_list = [1, 30, 50]

def load_pg(path, ind_list):
    f_list, tau_list, opt_ind_list = [], [], []
    fids_by_iter = np.loadtxt(path + 'fids_by_iters_1to50.txt')
    for ind in ind_list:
        f = np.loadtxt(path + f'fidelities_{ind}.txt')
        taus = np.loadtxt(path + f'taus_{ind}.txt') / 1e-9 * 2
        f_list.append(f)
        tau_list.append(taus)
        opt_ind_list.append(np.argmax(f))
    f_pg_list = [f_list[0]]
    for i in range(len(ind_list) - 1):
        f_pg_list.append(f_list[i+1] / fids_by_iter[ind_list[i+1] - 2])
    return tau_list, f_pg_list, opt_ind_list

# load data
tau_cold, fpg_cold, opt_cold = load_pg('data/fidelities/T1/n50/om_2.4_15/', ind_list)
tau_hot,  fpg_hot,  opt_hot  = load_pg('data/fidelities/T5/n200/om_2.4_15/', ind_list)

fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)

cold_color = '#1f77b4'
hot_color  = '#d62728'
alphas = [0.73, 0.88, 1.0][::-1]
# alphas = [1.0, 1.0, 1.0]
marker = 'o'

# common x-range info
xmin_all = min(min(t[0] for t in tau_cold), min(t[0] for t in tau_hot))
xmax_all = max(max(t[-1] for t in tau_cold), max(t[-1] for t in tau_hot))

# label positions (shared for both temperatures)
x_label1_cold = 0.525 * (xmax_all - xmin_all) + xmin_all
x_label1_hot = 0.6 * (xmax_all - xmin_all) + xmin_all
x_label30_cold = 0.71 * (xmax_all - xmin_all) + xmin_all  # N = 30
x_label50_cold = 0.75 * (xmax_all - xmin_all) + xmin_all  # N = 50
x_label30_hot = 0.79 * (xmax_all - xmin_all) + xmin_all  # N = 30
x_label50_hot = 0.75 * (xmax_all - xmin_all) + xmin_all  # N = 50
angles_cold = [0, 0, 0]
angles_hot = [0, 0, 0]
# angles_hot = [0, -9, -16]
gap_width = 25.0  # ns

def plot_curve_with_label(x, y, x_label, color_curve, color_text, angle, alpha, text):
    x = np.asarray(x); y = np.asarray(y)

    j0 = np.argmin(np.abs(x - x_label))
    x0, y0 = x[j0], y[j0]

    mask_left  = x <= x0 - gap_width/2
    mask_right = x >= x0 + gap_width/2

    if mask_left.sum() > 1:
        ax.plot(x[mask_left], y[mask_left],
                color=color_curve, lw=0.9, alpha=alpha)
    if mask_right.sum() > 1:
        ax.plot(x[mask_right], y[mask_right],
                color=color_curve, lw=0.9, alpha=alpha)

    ax.text(
        x0, y0,
        text,
        ha="center", va="center",
        color=color_text,
        fontsize=8,
        rotation=angle,
        rotation_mode='anchor',
        bbox=dict(boxstyle="square,pad=0.1",
                  fc="white", ec="none", alpha=1.0)
    )

# ----- COLD curves -----
for i, N in enumerate(ind_list):
    x = tau_cold[i]; y = fpg_cold[i]

    # optimal marker
    # x_opt = x[opt_cold[i]]; y_opt = y[opt_cold[i]]
    # ax.plot(x_opt, y_opt, linestyle='None', marker=marker,
    #         ms=4.0, mew=0.8, mfc='white', mec=cold_color,
    #         alpha=alphas[i])

    if N == ind_list[0]:
        plot_curve_with_label(x, y, x_label1_hot, cold_color, cold_color, angles_cold[0], alphas[i], r"")
        plot_curve_with_label(x, y, x_label1_cold, cold_color, cold_color, angles_cold[0], alphas[i], r"$1$")
    elif N == ind_list[1]:
        plot_curve_with_label(x, y, x_label30_cold, cold_color, cold_color, angles_cold[1], alphas[i], r"$30$")
    elif N == ind_list[2]:
        plot_curve_with_label(x, y, x_label50_cold, cold_color, cold_color, angles_cold[2], alphas[i], r"$50$")

# ----- HOT curves -----
for i, N in enumerate(ind_list):
    x = tau_hot[i]; y = fpg_hot[i]

    # x_opt = x[opt_hot[i]]; y_opt = y[opt_hot[i]]
    # ax.plot(x_opt, y_opt, linestyle='None', marker=marker,
    #         ms=4.0, mew=0.8, mfc='white', mec=hot_color,
    #         alpha=alphas[i])

    if N == ind_list[0]:
        plot_curve_with_label(x, y, x_label1_cold, hot_color, hot_color, angles_hot[0], alphas[i], r"")
        plot_curve_with_label(x, y, x_label1_hot, hot_color, hot_color, angles_hot[0], alphas[i], r"$1$")
    elif N == ind_list[1]:
        plot_curve_with_label(x, y, x_label30_hot, hot_color, hot_color, angles_hot[1], alphas[i], r"$30$")
    elif N == ind_list[2]:
        plot_curve_with_label(x, y, x_label50_hot, hot_color, hot_color, angles_hot[2], alphas[i], r"$50$")

# ----- axes & legend -----
ax.set_xlabel(r'$2\tau$ (ns)')
ax.set_ylabel(r'$f_N$')
ax.set_ylim(0.9805, 1.002)

xmin = xmin_all
xmax = xmax_all
margin = 0.03 * (xmax - xmin)
ax.set_xlim(xmin - margin, xmax + margin)

ax.set_yticks(np.linspace(0.985, 1.000, 4))
ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.8)
ax.tick_params(direction="in", which="both", top=True, right=True)


from matplotlib.lines import Line2D

ax.legend(
    [plt.Line2D([], [], color=cold_color, lw=0.9, alpha=0.0),
     plt.Line2D([], [], color=hot_color,  lw=0.9, alpha=0.0)],
    [r'$T = 1~\mu\mathrm{K}$', r'$T = 5~\mu\mathrm{K}$'],
    loc="lower left", frameon=True, bbox_to_anchor=(0.21, 0.003),
    handlelength=2.0
)

# fig.savefig('data/figure_cold_hot.png',
#             bbox_inches="tight", facecolor="white")
# fig.savefig('data/figure_cold_hot.pdf',
#             bbox_inches="tight", facecolor="white")
# fig.savefig('data/figure_cold_hot.svg',
#             bbox_inches="tight", facecolor="white")
plt.show()