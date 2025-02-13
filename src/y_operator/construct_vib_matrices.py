import numpy as np
from src.y_operator.params import HBAR, OM_small, M, Q

# TODO: check all constants here
# Number of basis oscillator vectors
n = 100

A_vib = np.eye(n, dtype=complex)

nu = np.arange(0, n)

const_p1 = -1j * np.sqrt(HBAR * OM_small * M / 2)
p1_1 = np.diag(np.sqrt(nu[1::]), 1)
p1_2 = np.diag(np.sqrt(nu[1::]), -1)
p1 = const_p1 * (p1_1 - p1_2)

const_p2 = HBAR * OM_small * M / 2
p2_1 = np.diag(2 * nu + 1)
p2_2 = np.diag(np.sqrt(nu[2::] * (nu[2::] - 1)), 2)
p2_3 = np.diag(np.sqrt(nu[2::] * (nu[2::] - 1)), -2)
p2 = const_p2 * (p2_1 - p2_2 - p2_3)

V1_vib = p2
V2_vib = p2 + 2 * p1 * HBAR * Q + (HBAR * Q) ** 2 * np.eye(n, dtype=complex)

W0z_vib = p1_1 + p1_2
