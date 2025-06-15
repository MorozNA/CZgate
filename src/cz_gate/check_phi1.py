import numpy as np
from calc_tools import get_U, calc_tau, calc_xi, calc_delta

om = 1
delta = 0.37737091627033376
tau = calc_tau(delta, om)
xi = calc_xi(delta, om, tau)

Ubb1, Urb1 = get_U(delta, om, 0.0, tau)[0, 0], get_U(delta, om, 0.0, tau)[1, 0]
Ubb2, Ubr2 = get_U(delta, om, xi, tau)[0, 0], get_U(delta, om, xi, tau)[0, 1]

e_phi1 = Ubb2 * Ubb1 + Ubr2 * Urb1
phi1 = np.angle(e_phi1)
phi1_other = xi + np.pi
phi2 = delta * tau

if phi1 < 0:
    phi1 += 2 * np.pi
if phi1_other < 0:
    phi1_other += 2 * np.pi
if phi2 < 0:
    phi2 += 2 * np.pi

print(phi1 - phi1_other)
