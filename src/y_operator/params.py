import numpy as np

OM = 2 * np.pi * 3.5e6
# xi2 = 2.380762
xi2 = 0.760829
tau = 4.292682 / OM
delta = 0.377371 * OM
A_INT_CONSTANT = 0.0

OM_small = 2 * np.pi * 7.158e3
HBAR = 1.05457159682e-34  # J \cdot s, or 6.582 118 89(26) eV \cdot s
M = 1.4431606011e-25  # kg
Q = 0.0  # TODO: make changable
V_INT_CONSTANT = 1.0


p0z = 1e-27  # kg * m / s
Z_ast = 1e-5  # m (10 micrometers)
W_INT_CONSTANT = 0.0
