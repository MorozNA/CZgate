import matplotlib.pyplot as plt
import numpy as np

# Set a LaTeX-like font (no need for rc('text', usetex=True))
plt.rcParams["font.family"] = "serif"  # Uses default serif font (like Times New Roman)
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # LaTeX-style math fonts

# Load your saved fidelity data
fidelity_iter1_temp0 = np.load('temp0/data/temp0_fidelity_iter1.npy')
fidelity_iter1_temp1 = np.load('temp1/data/temp1_fidelity_iter1.npy')
fidelity_iter1_temp3 = np.load('temp3/data/temp3_fidelity_iter1.npy')
fidelity_iter1_temp5 = np.load('temp5/data/temp5_fidelity_iter1.npy')
tau_array = np.load('temp0/data/temp0_tau.npy')

# Create x-axis values (tau_array values)
x = tau_array * 1e9  # Convert to nanoseconds for plotting
print(x[-1])

plt.figure(figsize=(8, 6), dpi=100)

# Plot each fidelity curve with different colors and styles
# plt.plot(x, fidelity_iter1_temp0, 'r', linewidth=2, label=r'$T \approx 0 K$')
plt.plot(x, fidelity_iter1_temp1, 'g', linewidth=2, label=r'$T = 1 \mu K$')
plt.plot(x, fidelity_iter1_temp3, 'b', linewidth=2, label=r'$T = 3 \mu K$')
plt.plot(x, fidelity_iter1_temp5, 'y', linewidth=2, label=r'$T = 5 \mu K$')
# used colors: r, g, b, y, k or '#d62728' '#2ca02c' '#1f77b4' '#ff7f0e' '#9467bd'
# plt.plot(x, fidelity_iter10, 'r--', linewidth=2, label='After 10 iterations')
# plt.plot(x, fidelity_iter25, 'g-.', linewidth=2, label='After 25 iterations')
# plt.plot(x, fidelity_iter50, 'm:', linewidth=2, label='After 50 iterations')

# Add horizontal line at fidelity=1 for reference
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3, label='Perfect fidelity')

# Customize the plot
# plt.title(r'Fidelity vs Gate Time ($2\tau$) at 0 µK', fontsize=14, pad=20)
# plt.xlabel(r'$2 \cdot τ$ (ns)', fontsize=12)
# plt.ylabel('Fidelity', fontsize=12)
plt.title(r"Fidelity vs Gate Time at 0 $\mu$K", fontsize=14, pad=20)
plt.xlabel(r"Gate time $2\tau$ (ns)", fontsize=12)
plt.ylabel(r"Fidelity", fontsize=12)  # \mathcal works without LaTeX!
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Set axis limits
plt.xlim(150, 1450)  # Matching your tau_array range in ns
# plt.ylim(0.98, 1.0025)

# Add legend
plt.legend(loc='lower right', fontsize=10, framealpha=1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure (optional)
plt.savefig('fidelity_vs_tau.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()