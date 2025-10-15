import matplotlib.pyplot as plt
import numpy as np

# Set a LaTeX-like font (no need for rc('text', usetex=True))
plt.rcParams["font.family"] = "serif"  # Uses default serif font (like Times New Roman)
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # LaTeX-style math fonts

# Load your saved fidelity data_W
# fidelity_30 = np.load('fidelity_iter1_plot2023_30MHz.npy')
# fidelity_40 = np.load('fidelity_iter1_plot2023_40MHz.npy')
# fidelity_50 = np.load('fidelity_iter1_plot2023_50MHz.npy')
# fidelity_60 = np.load('fidelity_iter1_plot2023_60MHz.npy')
f0 = np.load('temp0_fidelity_iter1.npy')
f1 = np.load('temp1_fidelity_iter1.npy')
f3 = np.load('temp3_fidelity_iter1.npy')
f5 = np.load('temp5_fidelity_iter1.npy')
x = np.load("temp5_tau.npy")
#fidelity_test = np.load('fidelity_iter1_plot2023_40MHz_test.npy')
om_array = np.load('om_array_plot2023_40MHz_test.npy')

# Create x-axis values (tau_array values)
# x = om_array / 1e6 / 2 / np.pi  # Convert to nanoseconds for plotting
print(x[-1])

plt.figure(figsize=(8, 6), dpi=100)

# Plot each fidelity curve with different colors and styles
plt.plot(x, f0, 'r', linewidth=2, label=r'$\delta_R$ = 30 MHz')
plt.plot(x, f1, 'r', linewidth=2, label=r'$\delta_R$ = 30 MHz')
plt.plot(x, f3, 'r', linewidth=2, label=r'$\delta_R$ = 30 MHz')
plt.plot(x, f5, 'r', linewidth=2, label=r'$\delta_R$ = 30 MHz')
# plt.plot(x, fidelity_test, 'r', linewidth=2, label=r'$\delta_R$ = 30 MHz')
# plt.plot(x, fidelity_30, 'r', linewidth=2, label=r'$\delta_R$ = 30 MHz')
# plt.plot(x, fidelity_40, 'g', linewidth=2, label=r'$\delta_R$ = 40 MHz')
# plt.plot(x, fidelity_50, 'b', linewidth=2, label=r'$\delta_R$ = 50 MHz')
# plt.plot(x, fidelity_60, 'y', linewidth=2, label=r'$\delta_R$ = 60 MHz')

# Add horizontal line at fidelity=1 for reference
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3, label='Perfect fidelity')

# Customize the plot
plt.title(r"Fidelity vs Gate Time at 0 $\mu$K", fontsize=14, pad=20)
plt.xlabel(r"Gate time $2\tau$ (ns)", fontsize=12)
plt.ylabel(r"Fidelity", fontsize=12)  # \mathcal works without LaTeX!
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Set axis limits
# plt.xlim(150, 1450)  # Matching your tau_array range in ns
# plt.ylim(0, 1.1)

# Add legend
plt.legend(loc='lower right', fontsize=10, framealpha=1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure (optional)
plt.savefig('fidelity_vs_tau.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()