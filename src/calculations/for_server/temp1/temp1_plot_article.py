import matplotlib.pyplot as plt
import numpy as np

# PRA-style figure settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.figsize': (3.37, 2.5),  # PRA column width (3.37 inches)
    'figure.dpi': 300,
    'lines.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'mathtext.fontset': 'stix',
})

# Load data
data_path = 'data/'
tau_array = np.load(data_path + 'temp1_tau.npy') * 1e9  # Convert to ns
iterations = [1, 5, 10, 15, 20, 25]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
fidelities = [np.load(f'{data_path}temp1_fidelity_iter{n}.npy') for n in iterations]

# Create figure
fig, ax = plt.subplots(figsize=(3.37, 2.5))  # Single column width for PRA

# Plot data
for n, fid, color in zip(iterations, fidelities, colors):
    ax.plot(tau_array, fid, color=color, label=f'{n} iter.')

# Reference line
ax.axhline(y=1, color='k', linestyle=':', alpha=0.5, linewidth=1)

# Axis settings
ax.set_xlim(150, 1450)
ax.set_ylim(0.7, 1.02)
ax.set_xlabel(r'Gate time $2\tau$ (ns)')
ax.set_ylabel('Fidelity')

# Legend
ax.legend(loc='lower right', frameon=True, framealpha=1)

# Tight layout and save
plt.tight_layout(pad=0.3)
plt.savefig('fidelity_vs_tau.pdf', format='pdf', bbox_inches='tight')
plt.close()