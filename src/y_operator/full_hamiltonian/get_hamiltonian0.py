import numpy as np
from src.y_operator.change_basis import change_basis


def get_H_simple(delta, om, xi):
    H = np.zeros((2, 2), dtype=complex)
    H[0, 0] = 0.0
    H[0, 1] = om / 2 * np.exp(1j * xi)
    H[1, 0] = H[0, 1].conj()
    H[1, 1] = -delta
    return H


def get_H_deltaR(delta, om, xi, delta_rydberg):
    H = np.zeros((3, 3), dtype=complex)
    H[0, 0] = 0
    H[0, 1] = (np.sqrt(2) * om) / 2 * np.exp(1j * xi)
    H[1, 0] = H[0, 1].conj()
    H[1, 1] = - delta
    H[1, 2] = (np.sqrt(2) * om) / 2 * np.exp(1j * xi)
    H[2, 1] = H[1, 2].conj()
    H[2, 2] = delta_rydberg - 2 * delta
    return H


def get_H0(delta, om, xi, delta_rydberg=None):
    use_deltaR = delta_rydberg is not None

    H = np.zeros((9, 9), dtype=complex)
    H[1:3, 1:3] = get_H_simple(delta, om, xi)
    H[3:5, 3:5] = get_H_simple(delta, om, xi)

    if use_deltaR:
        H_deltaR1 = get_H_deltaR(delta, om, xi, delta_rydberg)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        H[np.ix_(rows, cols)] = H_deltaR1
    else:
        H[5:7, 5:7] = get_H_simple(delta, np.sqrt(2) * om, xi)

    return change_basis(H)
