from src.y_operator.params import OM_small
import numpy as np

n = 100
# Constants in SI units
kB = 1.380649e-23  # Boltzmann constant in J/K
T = 1e-8  # Temperature in Kelvin (adjust as needed)
beta = 1 / (kB * T)
hbar = 1.0545718e-34  # Reduced Planck's constant in J·s
omega = OM_small  # Example frequency in rad/s (adjust if needed) # TODO: ask Leonid about it

# Vibrational energy levels
En = hbar * omega * (np.arange(n) + 0.5)  # Energy levels

# Normalize energy by kB*T before exponentiation to avoid underflow
E_normalized = En / (kB * T)
# print(hbar * omega / kB / T)
rho_T0 = np.diag(np.exp(-E_normalized))
rho_T0 /= np.trace(rho_T0)  # Normalize

# Print for verification
# print("Initial vibrational density matrix (rho_T0):")
# print(np.round(rho_T0, 2))
