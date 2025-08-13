import numpy as np
from src.y_operator_deltaR.full_hamiltonian.change_basis import change_basis


def get_H_simple(delta, om, xi):
    Hamiltonian = np.zeros((2, 2), dtype=complex)
    Hamiltonian[0, 0] = 0.0
    Hamiltonian[0, 1] = om / 2 * np.exp(1j * xi)
    Hamiltonian[1, 0] = Hamiltonian[0, 1].conj()
    Hamiltonian[1, 1] = -delta
    return Hamiltonian


def get_H_deltaR(delta, om, xi, delta_rydberg):
    Hamiltonian = np.zeros((3, 3), dtype=complex)
    Hamiltonian[0, 0] = 0
    Hamiltonian[0, 1] = (np.sqrt(2) * om) / 2 * np.exp(1j * xi)
    Hamiltonian[1, 0] = Hamiltonian[0, 1].conj()
    Hamiltonian[1, 1] = - delta
    Hamiltonian[1, 2] = (np.sqrt(2) * om) / 2 * np.exp(1j * xi)
    Hamiltonian[2, 1] = Hamiltonian[1, 2].conj()
    Hamiltonian[2, 2] = delta_rydberg - 2 * delta
    return Hamiltonian


def get_H0(delta, om, xi, delta_rydberg):
    Hamiltonian = np.zeros((9, 9), dtype=complex)
    Hamiltonian[1:3, 1:3] = get_H_simple(delta, om, xi)
    Hamiltonian[3:5, 3:5] = get_H_simple(delta, om, xi)

    H_deltaR1 = get_H_deltaR(delta, om, xi, delta_rydberg)
    rows = [5, 6, 8]  # Target rows in U1
    cols = [5, 6, 8]  # Target columns in U1
    Hamiltonian[np.ix_(rows, cols)] = H_deltaR1

    return change_basis(Hamiltonian)
