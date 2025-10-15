import numpy as np
from scipy.linalg import expm
from src.y_operator.full_hamiltonian.get_hamiltonian0 import get_H0
from src.y_operator.full_hamiltonian.get_hamiltonian_int import get_HA, get_HB


def get_evolution(delta, om, xi, tau, n, q):
    H01 = np.kron(get_H0(delta, om, 0.0), np.kron(np.eye(n), np.eye(n)))
    H02 = np.kron(get_H0(delta, om, xi), np.kron(np.eye(n), np.eye(n)))
    H1 = H01 + get_HA(om, tau, delta, 0.0, n, q) + get_HB(om, tau, delta, 0.0, n, q)
    H2 = H02 + get_HA(om, tau, delta, xi, n, q) + get_HB(om, tau, delta, xi, n, q)
    return expm(-1j * H2 * tau) @ expm(-1j * H1 * tau)

