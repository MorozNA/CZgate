import numpy as np

# EFFECT CONSTANTS
W_INT_CONSTANT = 1.0

# GLOBAL CONSTANTS
kB = 1.380649e-23  # Boltzmann constant in J/K
HBAR = 1.05457159682e-34  # J \cdot s, or 6.582 118 89(26) eV \cdot s
OM_small = 2 * np.pi * 7.158e3
M = 1.4431606011e-25  # kg, Rb atom mass

# W-matrix constants
lambd_1 = 795e-9
lambd_2 = 480e-9
w01 = 10e-6
w02 = 10e-6
z_r1 = np.pi * w01 ** 2 / lambd_1
z_r2 = np.pi * w02 ** 2 / lambd_2
p0z = np.sqrt(HBAR * M * OM_small / 2)  # kg * m / s; coeff '2' is saved here, but not used in momentum matrices
Z_ast = 2 * z_r1 * z_r2 / (z_r1 + z_r2)

z_ij_matrix = np.zeros((3, 3), dtype=complex)
z_ij_matrix[0, 0] = 1 / z_r1 ** 2
z_ij_matrix[1, 1] = 1 / z_r1 ** 2
z_ij_matrix[1, 2] = 1 / 2 / (z_r1 ** 2) + 1 / 2 / (z_r2 ** 2)
z_ij_matrix[2, 1] = 1 / 2 / (z_r1 ** 2) + 1 / 2 / (z_r2 ** 2)
z_ij_matrix[2, 2] = 1 / z_r2 ** 2
DELTA_b = DELTA_r = - 2 * np.pi * 2.5 * 1e6
DELTA_a = - 2 * np.pi * 0.7 * 1e6


x_ij_matrix = np.zeros((3, 3), dtype=complex)
x_ij_matrix[0, 0] = 1 / w01 ** 2
x_ij_matrix[1, 1] = 1 / w01 ** 2
x_ij_matrix[1, 2] = 1 / (w01 ** 2) + 1 / (w02 ** 2)
x_ij_matrix[2, 1] = 1 / (w01 ** 2) + 1 / (w02 ** 2)
x_ij_matrix[2, 2] = 1 / w01 ** 2

# ### INTIAL STATE PARAMS
# n = 100
