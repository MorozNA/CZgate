import numpy as np
import matplotlib.pyplot as plt
from src.calculations_no_leakage.plot_temperature.approximate_function import get_temperature
from src.calculations_no_leakage.for_server.get_rho_fun import get_rho_T0
from src.calculations_no_leakage.for_server.distribution_change.temp0_distr_init import n
from src.y_operator_deltaR.params import HBAR, OM_small

# PRA-style figure settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.figsize': (3.37, 2.5),  # PRA column width
    'figure.dpi': 300,
    'lines.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'mathtext.fontset': 'stix',
})

# Load and process data_W
rho_TA_list = np.abs(np.load('temp1_rho_TA_list.npy'))
rho_TB_list = np.abs(np.load('temp1_rho_TA_list.npy'))  # Note: Same file as TA?
temperatures = [get_temperature(np.diag(rho)) for rho in rho_TB_list]

# Setup
x = np.arange(len(rho_TB_list[0]))
energies = HBAR * OM_small * (x + 0.5)
selected_iters = [1, 10, 50, 200]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colorblind-friendly palette

# Create figure
fig, ax = plt.subplots(figsize=(3.37, 2.5))

# Plot distributions
for idx, i in enumerate(selected_iters):
    color = colors[idx]
    actual_probs = rho_TB_list[i].real / np.sum(rho_TB_list[i].real)
    fitted_probs = np.diag(get_rho_T0(temperatures[i], n))

    ax.plot(x, actual_probs, '-', color=color,
            label=fr'${i}$ iter. ($T_\mathrm{{fit}}={temperatures[i] * 1e6:.1f}\,\mu\mathrm{{K}}$)')
    ax.plot(x, fitted_probs, '--', color=color, linewidth=1)

# Formatting
ax.set(xlabel='Energy level $n$',
       ylabel='Probability (log)',
       yscale='log',
       xlim=(x[0], x[-1]),
       ylim=(1e-10, 0.3))
ax.legend(loc='upper right', frameon=True, framealpha=1)

plt.tight_layout(pad=0.3)
plt.savefig('fig1.pdf', format='pdf', bbox_inches='tight')
plt.close()