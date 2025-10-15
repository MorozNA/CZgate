import numpy as np
from src.y_operator.full_hamiltonian.change_basis import change_basis


def get_H_simple(delta, om, xi):
    Hamiltonian = np.zeros((2, 2), dtype=complex)
    Hamiltonian[0, 0] = 0.0
    Hamiltonian[0, 1] = om / 2 * np.exp(1j * xi)
    Hamiltonian[1, 0] = Hamiltonian[0, 1].conj()
    Hamiltonian[1, 1] = -delta
    return Hamiltonian


def get_H0(delta, om, xi):
    Hamiltonian = np.zeros((9, 9), dtype=complex)
    Hamiltonian[1:3, 1:3] = get_H_simple(delta, om, xi)
    Hamiltonian[3:5, 3:5] = get_H_simple(delta, om, xi)
    Hamiltonian[5:7, 5:7] = get_H_simple(delta, np.sqrt(2) * om, xi)
    return change_basis(Hamiltonian)
