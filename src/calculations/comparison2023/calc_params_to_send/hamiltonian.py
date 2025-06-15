import numpy as np
from scipy.linalg import expm


def get_Hamiltonian(om, delta, xi):
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0, 0] = 0.0
    hamiltonian[0, 1] = om / 2 * np.exp(1j * xi)
    hamiltonian[1, 0] = om / 2 * np.exp(-1j * xi)
    hamiltonian[1, 1] = -delta
    return hamiltonian


def get_Hamiltonian_leakage(om, delta, xi, delta_rydberg):
    hamiltonian = np.zeros((3, 3), dtype=complex)
    hamiltonian[0, 0] = 0
    hamiltonian[0, 1] = (np.sqrt(2) * om) / 2 * np.exp(1j * xi)
    hamiltonian[1, 0] = hamiltonian[0, 1].conj()
    hamiltonian[1, 1] = - delta
    hamiltonian[1, 2] = (np.sqrt(2) * om) / 2 * np.exp(1j * xi)
    hamiltonian[2, 1] = hamiltonian[1, 2].conj()
    hamiltonian[2, 2] = delta_rydberg
    return hamiltonian


def get_FullHamiltonian(om, delta, xi, delta_rydberg):
    # Constructing block-diagonal hamiltonian
    hamiltonian = np.zeros((9, 9), dtype=complex)
    hamiltonian[1:3, 1:3] = get_Hamiltonian(om, delta, xi)
    hamiltonian[3:5, 3:5] = get_Hamiltonian(om, delta, xi)
    idx = [5, 6, 8]  # Target rows and cols
    hamiltonian[np.ix_(idx, idx)] = get_Hamiltonian_leakage(om, delta, xi, delta_rydberg)
    return hamiltonian


def get_evolution(om, delta, xi, delta_rydberg, t):
    delta_renorm = delta + (om ** 2) / (2 * delta_rydberg)
    tau = 2 * np.pi / np.sqrt(2 * om ** 2 + delta_renorm ** 2)

    if t <= tau:
        return expm(-1j * get_FullHamiltonian(om, delta, 0.0, delta_rydberg) * t)
    else:
        U1 = expm(-1j * get_FullHamiltonian(om, delta, 0.0, delta_rydberg) * tau)
        U2 = expm(-1j * get_FullHamiltonian(om, delta, xi, delta_rydberg) * (t - tau))
        return U2 @ U1
