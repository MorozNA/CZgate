import matplotlib.pyplot as plt
import numpy as np

# Load your saved fidelity data_W
fidelity_iter1 = np.load('temp_tst_fidelity_iter1.npy')
fidelity_iter5 = np.load('temp_tst_fidelity_iter5.npy')
fidelity_iter10 = np.load('temp_tst_fidelity_iter10.npy')
fidelity_iter15 = np.load('temp_tst_fidelity_iter15.npy')
fidelity_iter20 = np.load('temp_tst_fidelity_iter20.npy')
tau_array = np.load('temp_tst_tau.npy')

# Create x-axis values (tau_array values)
x = tau_array * 1e9  # Convert to nanoseconds for plotting

plt.figure(figsize=(10, 6), dpi=100)

# Plot each fidelity curve with different colors and styles
plt.plot(x, fidelity_iter1, 'r-', linewidth=2, label='After 1 iteration')
plt.plot(x, fidelity_iter5, 'g--', linewidth=2, label='After 5 iterations')
# plt.plot(x, fidelity_iter10, 'b-.', linewidth=2, label='After 10 iterations')
# plt.plot(x, fidelity_iter15, 'y--', linewidth=2, label='After 15 iterations')
# plt.plot(x, fidelity_iter20, 'p--', linewidth=2, label='After 20 iterations')


# Add horizontal line at fidelity=1 for reference
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3, label='Perfect fidelity')

# Customize the plot
plt.title('Fidelity vs Pulse Duration (τ)', fontsize=14, pad=20)
plt.xlabel('Pulse duration τ (ns)', fontsize=12)
plt.ylabel('Fidelity', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Set axis limits
plt.xlim(150, 1650)  # Matching your tau_array range in ns
plt.ylim(0, 1.1)

# Add legend
plt.legend(loc='lower right', fontsize=10, framealpha=1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure (optional)
# plt.savefig('fidelity_vs_tau.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()