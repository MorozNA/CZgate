import numpy as np
import matplotlib.pyplot as plt
from src.calculations.plot_temperature.approximate_function import get_temperature
from src.calculations.for_server.get_rho_fun import get_rho_T0
from src.calculations.for_server.distribution_change.temp0_distr_init import n
from src.y_operator.params import HBAR, OM_small

# Set a LaTeX-like font (no need for rc('text', usetex=True))
plt.rcParams["font.family"] = "serif"  # Uses default serif font (like Times New Roman)
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # LaTeX-style math fonts

# Load data
rho_TA_list = np.load('temp1_rho_TA_list.npy')
rho_TB_list = np.load('temp1_rho_TA_list.npy')
rho_TA_list = abs(rho_TA_list)
rho_TB_list = abs(rho_TB_list)

# Calculate temperatures
temperatures = [get_temperature(np.diag(rho)) for rho in rho_TB_list]

# Iterations to plot
selected_iters = [1, 10, 50, 200]

# X-axis setup
x = np.arange(len(rho_TB_list[0]))
energies = HBAR * OM_small * (np.arange(len(x)) + 0.5)

# Plot
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_title('Probability distribution', fontsize=14, pad=20)
ax.set_xlabel('Energy Level Number', fontsize=12)
ax.set_ylabel('Probability (log)', fontsize=12)
ax.set_yscale('log')
ax.set_xlim(x[0], x[-1])
ax.set_ylim(1e-10, 0.3)

# Get a colormap
# cmap = plt.get_cmap('tab10')
colors = ['red', 'green', 'blue', 'orange']

# Plot actual and fitted distributions for each selected iteration
for idx, i in enumerate(selected_iters):
    #color = cmap(idx % 10)
    color = colors[idx]

    actual_rho = rho_TB_list[i]
    actual_probs = actual_rho.real / np.sum(actual_rho.real)

    temp = temperatures[i]
    fitted_probs = np.diag(get_rho_T0(temp, n))

    ax.plot(x, actual_probs, label=f'{i} iterations', linestyle='-', color=color)
    ax.plot(x, fitted_probs, label=f'fit with T={temp * 1e6:.2f}$\mu$K)', linestyle='--', color=color)

plt.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

# Save the figure (optional)
plt.savefig('prob_1muK.png', dpi=300, bbox_inches='tight')

plt.show()
