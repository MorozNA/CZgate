from main_code import iterations, fidelity0, fidelity1, fidelity2, fidelity3
import matplotlib.pyplot as plt
import numpy as np

# Assuming you've already run your iteration code and have the fidelity lists

# Create x-axis values (each iteration step)
# Since you append twice per iteration (first and second iteration), we need double the points
x = np.arange(1, iterations + 1)  # Shows half-steps for each iteration

plt.figure(figsize=(10, 6), dpi=100)

# Plot each fidelity curve with different colors and styles
plt.plot(x, fidelity0, 'b-', linewidth=2, markersize=6, label=r'$T_0 \approx 0K$')
plt.plot(x, fidelity1, 'r--', linewidth=2, markersize=6, label=r'$T_1 = 1 \mu K$')
plt.plot(x, fidelity2, 'g-.', linewidth=2, markersize=6, label=r'$T_2 = 5 \mu K$')
plt.plot(x, fidelity3, 'm:', linewidth=2, markersize=6, label=r'$T_3 = 10 \mu K$')

# Add horizontal line at fidelity=1 for reference
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3, label='1.0 fidelity line')

# Customize the plot
plt.title('Fidelity vs Iteration Number', fontsize=14, pad=20)
plt.xlabel('Iteration Number', fontsize=12)
plt.ylabel('Fidelity', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.xticks(np.arange(0, 2 * (iterations+1), 1))  # Show integer iteration numbers
# plt.yticks(np.linspace(0, 1.1, 12))  # Fidelity typically between 0 and 1

# Set axis limits
plt.xlim(0, iterations + 2)
plt.ylim(0, 1.1)

# Add legend
plt.legend(loc='lower right', fontsize=10, framealpha=1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure (optional)
# plt.savefig('fidelity_vs_iterations.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()