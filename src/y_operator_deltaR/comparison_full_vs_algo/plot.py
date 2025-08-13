import numpy as np
from src.y_operator_deltaR.comparison_full_vs_algo.comparison import fidelity, fidelity_spin, fidelity_vib, iterations
from src.y_operator_deltaR.comparison_full_vs_algo.comparison import purity_spin_full, purity_spin_alg, fidelity_decomp


x_step = iterations // 10
path = 'data5/'


import matplotlib.pyplot as plt

# Create the plot with improved styling
plt.figure(figsize=(8, 5))  # Larger figure size
plt.plot(np.arange(iterations) + 1, fidelity, 'o-',
         color='#1f77b4',  # Use a professional blue color
         markersize=6,     # Slightly larger markers
         linewidth=1.5,     # Slightly thicker line
         markerfacecolor='white',  # White fill for markers
         markeredgewidth=1.5)      # Marker edge thickness

# Labels and title with improved styling
plt.xlabel('Iteration', fontsize=12, labelpad=10)
plt.ylabel('Fidelity', fontsize=12, labelpad=10)
plt.title('Fidelity between full density matrices', fontsize=14, pad=20)

# Ensure x-ticks are integers
plt.xticks(np.arange(1, iterations + 1, x_step), fontsize=10)
plt.yticks(fontsize=10)

# Grid and layout improvements
plt.grid(True, linestyle='--', alpha=0.7)  # Dashed grid lines
plt.tight_layout()  # Adjust layout to prevent label clipping


# After all your plotting commands, before plt.show()
plt.savefig(path + 'fidelity_vs_iteration.png',
            dpi=300,                   # High resolution (300-600 for publications)
            bbox_inches='tight',        # Remove extra whitespace
            transparent=False)

# Show plot
plt.show()


# SECOND PLOT ____________________________________________________________
# Create the plot with improved styling
plt.figure(figsize=(8, 5))  # Larger figure size
plt.plot(np.arange(iterations) + 1, fidelity_spin, 'o-',
         color='#1f77b4',  # Use a professional blue color
         markersize=6,     # Slightly larger markers
         linewidth=1.5,     # Slightly thicker line
         markerfacecolor='white',  # White fill for markers
         markeredgewidth=1.5)      # Marker edge thickness

# Labels and title with improved styling
plt.xlabel('Iteration', fontsize=12, labelpad=10)
plt.ylabel('Fidelity', fontsize=12, labelpad=10)
plt.title('Fidelity between spin density matrices', fontsize=14, pad=20)

# Ensure x-ticks are integers
plt.xticks(np.arange(1, iterations + 1, x_step), fontsize=10)
plt.yticks(fontsize=10)

# Grid and layout improvements
plt.grid(True, linestyle='--', alpha=0.7)  # Dashed grid lines
plt.tight_layout()  # Adjust layout to prevent label clipping


plt.savefig(path + 'fidelity_spin_vs_iteration.png',
            dpi=300,                   # High resolution (300-600 for publications)
            bbox_inches='tight',        # Remove extra whitespace
            transparent=False)

# Show plot
plt.show()


# 'SECOND' SECOND PLOT ____________________________________________________________
# Create the plot with improved styling
plt.figure(figsize=(8, 5))  # Larger figure size
plt.plot(np.arange(iterations) + 1, fidelity_vib, 'o-',
         color='#1f77b4',  # Use a professional blue color
         markersize=6,     # Slightly larger markers
         linewidth=1.5,     # Slightly thicker line
         markerfacecolor='white',  # White fill for markers
         markeredgewidth=1.5)      # Marker edge thickness

# Labels and title with improved styling
plt.xlabel('Iteration', fontsize=12, labelpad=10)
plt.ylabel('Fidelity', fontsize=12, labelpad=10)
plt.title('Fidelity between vibrational density matrices', fontsize=14, pad=20)

# Ensure x-ticks are integers
plt.xticks(np.arange(1, iterations + 1, x_step), fontsize=10)
plt.yticks(fontsize=10)

# Grid and layout improvements
plt.grid(True, linestyle='--', alpha=0.7)  # Dashed grid lines
plt.tight_layout()  # Adjust layout to prevent label clipping


plt.savefig(path + 'fidelity_vib_vs_iteration.png',
            dpi=300,                   # High resolution (300-600 for publications)
            bbox_inches='tight',        # Remove extra whitespace
            transparent=False)

# Show plot
plt.show()


# ''SECOND'' 'SECOND' SECOND PLOT ____________________________________________________________
# Create the plot with improved styling
plt.figure(figsize=(8, 5))  # Larger figure size
plt.plot(np.arange(iterations) + 1, fidelity_decomp, 'o-',
         color='#1f77b4',  # Use a professional blue color
         markersize=6,     # Slightly larger markers
         linewidth=1.5,     # Slightly thicker line
         markerfacecolor='white',  # White fill for markers
         markeredgewidth=1.5)      # Marker edge thickness

# Labels and title with improved styling
plt.xlabel('Iteration', fontsize=12, labelpad=10)
plt.ylabel('Fidelity', fontsize=12, labelpad=10)
plt.title('Fidelity between decomposed density matrices', fontsize=14, pad=20)

# Ensure x-ticks are integers
plt.xticks(np.arange(1, iterations + 1, x_step), fontsize=10)
plt.yticks(fontsize=10)

# Grid and layout improvements
plt.grid(True, linestyle='--', alpha=0.7)  # Dashed grid lines
plt.tight_layout()  # Adjust layout to prevent label clipping


plt.savefig(path + 'fidelity_decomp_vs_iteration.png',
            dpi=300,                   # High resolution (300-600 for publications)
            bbox_inches='tight',        # Remove extra whitespace
            transparent=False)

# Show plot
plt.show()


# THIRD PLOT ____________________________________________________________
# Create the plot with improved styling
plt.figure(figsize=(8, 5))  # Larger figure size
plt.plot(np.arange(iterations) + 1, purity_spin_full, 'o-',
         color='#1f77b4',  # Use a professional blue color
         markersize=6,     # Slightly larger markers
         linewidth=1.5,     # Slightly thicker line
         markerfacecolor='white',  # White fill for markers
         markeredgewidth=1.5,      # Marker edge thickness
         label='full')

plt.plot(np.arange(iterations) + 1, purity_spin_alg, 'o-',
         color='red',  # Use a professional blue color
         markersize=6,     # Slightly larger markers
         linewidth=1.5,     # Slightly thicker line
         markerfacecolor='white',  # White fill for markers
         markeredgewidth=1.5,      # Marker edge thickness
         label='alg')

# Labels and title with improved styling
plt.xlabel('Iteration', fontsize=12, labelpad=10)
plt.ylabel('Purity', fontsize=12, labelpad=10)
# plt.title('Entanglement between spatial degrees of freedom', fontsize=14, pad=20)

# Ensure x-ticks are integers
plt.xticks(np.arange(1, iterations + 1, x_step), fontsize=10)
plt.yticks(fontsize=10)

# Grid and layout improvements
plt.grid(True, linestyle='--', alpha=0.7)  # Dashed grid lines
plt.tight_layout()  # Adjust layout to prevent label clipping
plt.legend()


plt.savefig(path + 'purity_spin.png',
            dpi=300,                   # High resolution (300-600 for publications)
            bbox_inches='tight',        # Remove extra whitespace
            transparent=False)

# Show plot
plt.show()