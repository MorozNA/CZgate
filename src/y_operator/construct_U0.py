import numpy as np
from src.y_operator.params import OM, xi2, tau, delta


def swap_basis(matrix, i, j):
    """Swap rows and columns i and j of a square matrix."""
    matrix[[i, j], :] = matrix[[j, i], :]  # Swap rows
    matrix[:, [i, j]] = matrix[:, [j, i]]  # Swap columns
    return matrix


def apply_hadamard(matrix, i, j):
    """
    Apply a Hadamard gate to two specific basis vectors (rows and columns).

    Parameters:
        matrix: np.ndarray - The unitary matrix.
        i: int - Index of the first basis vector.
        j: int - Index of the second basis vector.
    """
    # Define the Hadamard matrix
    Hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)

    # Apply Hadamard to rows i and j
    submatrix = matrix[[i, j], :].copy()  # Extract rows i and j
    matrix[[i, j], :] = Hadamard @ submatrix  # Apply Hadamard to rows

    # Apply Hadamard to columns i and j
    submatrix = matrix[:, [i, j]].copy()  # Extract columns i and j
    matrix[:, [i, j]] = submatrix @ Hadamard.conj().T  # Apply Hadamard to columns

    return matrix


def change_basis(matrix):
    """Apply Hadamard and swap transformations to a unitary matrix."""
    if not np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0])):
        raise ValueError("Input matrix is not unitary!")

    # Apply Hadamard transformation to submatrix [6:8, 6:8]
    matrix = apply_hadamard(matrix, 6, 7)

    # Swap basis vectors
    matrix = swap_basis(matrix, 4, 5)
    matrix = swap_basis(matrix, 5, 6)

    # Verify unitarity after transformations
    if not np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0])):
        raise ValueError("Resulting matrix is not unitary!")

    return matrix


def get_U(delta, omega, xi, t):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    Ubb = (np.cos(om_0 * t / 2) + 1j * delta / om_0 * np.sin(om_0 * t / 2)) * np.exp(-1j * delta * t / 2)
    Ubr = 1j * omega / om_0 * np.sin(om_0 * t / 2) * np.exp(1j * xi - 1j * delta * t / 2)
    Urb = 1j * omega / om_0 * np.sin(om_0 * t / 2) * np.exp(-1j * xi + 1j * delta * t / 2)
    Urr = (np.cos(om_0 * t / 2) - 1j * delta / om_0 * np.sin(om_0 * t / 2)) * np.exp(1j * delta * t / 2)
    U = np.array([[Ubb, Ubr], [Urb, Urr]], dtype=complex)
    return U


def construct_U0(t):
    if t <= tau:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, OM, 0.0, t)
        U1[3:5, 3:5] = get_U(delta, OM, 0.0, t)
        U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * OM, 0.0, t)
        return change_basis(U1)
    else:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, OM, 0.0, tau)
        U1[3:5, 3:5] = get_U(delta, OM, 0.0, tau)
        U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * OM, 0.0, tau)

        U2 = np.eye(9, dtype=complex)
        U2[1:3, 1:3] = get_U(delta, OM, xi2, t - tau)
        U2[3:5, 3:5] = get_U(delta, OM, xi2, t - tau)
        U2[5:7, 5:7] = get_U(delta, np.sqrt(2) * OM, xi2, t - tau)
        return change_basis(U2 @ U1)