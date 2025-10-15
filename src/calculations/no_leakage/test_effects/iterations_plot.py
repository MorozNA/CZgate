import numpy as np
from matplotlib import pyplot as plt


path = 'data_QW/temp05/'
effect_name = r'Both effects, '
# temp_name = r'$T = 0 \; K$'
temp_name = r'$T = 0.5 \; \mu K$'

fidelity0_alg = np.loadtxt(path + 'fidelity0_alg.txt')
fidelity0_full = np.loadtxt(path + 'fidelity0_full.txt')
fidelity0_full_decomp = np.loadtxt(path + 'fidelity0_full_decomp.txt')
purity = np.loadtxt(path + 'purity.txt')
iterations = len(fidelity0_alg)

plt.figure(figsize=(10, 6))

plt.plot(range(iterations), 1 - np.array(fidelity0_alg), color='green', linewidth=2, label=r'$\rho^{(S)}_{algorithm}$')
plt.plot(range(iterations), 1 - np.array(fidelity0_full), color='purple', linewidth=2, label=r'$\rho^{(S)}_{full}$')
plt.plot(range(iterations), 1 - np.array(fidelity0_full_decomp), color='purple', linestyle='dashed', linewidth=2, label=r'$\rho^{(S)}_{full}$ with decomposition')

plt.title(r'$1 - F(\rho^{(S)}, \rho^{(S)}_{ideal}$), ' + effect_name + temp_name , fontsize=16)
plt.ylabel(r'Infidelity', fontsize=14)
plt.xlabel(r'Num. of iter.', fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the figure in high resolution (adjust parameters as needed)
plt.savefig(
    path + 'infidelity_plot.png',  # Filename (supports .png, .pdf, .svg, .jpg)
    dpi=300,               # Dots per inch (higher = better resolution)
    bbox_inches='tight',   # Removes extra whitespace
    format='png',          # File format ('png', 'pdf', 'svg', 'jpeg')
    transparent=False      # Set to True if you want a transparent background
)

plt.show()


plt.figure(figsize=(10, 6))
plt.plot(range(iterations), 1 - purity, color='purple', linewidth=2)
plt.title(r'Purity of $\rho^{(S)}_{full}$, ' + temp_name, fontsize=16)
plt.ylabel(r'Purity', fontsize=14)
plt.xlabel(r'Num. of iter.', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig(
    path + 'purity.png',  # Filename (supports .png, .pdf, .svg, .jpg)
    dpi=300,               # Dots per inch (higher = better resolution)
    bbox_inches='tight',   # Removes extra whitespace
    format='png',          # File format ('png', 'pdf', 'svg', 'jpeg')
    transparent=False      # Set to True if you want a transparent background
)

plt.show()