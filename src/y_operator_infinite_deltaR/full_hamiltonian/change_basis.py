import numpy as np


def swap_basis(matrix, i, j):
    """Swap rows and columns i and j of a square matrix."""
    perm_matrix = np.eye(matrix.shape[0], dtype=complex)
    perm_matrix[[i, j], :] = perm_matrix[[j, i], :]
    return perm_matrix @ matrix @ perm_matrix.T


def apply_hadamard(matrix, i, j):
    """ Applies a Hadamard gate to basis vectors `i` and `j` using a transformation matrix. """
    identity = np.eye(matrix.shape[0], dtype=complex)
    hadamard_matrix = np.eye(matrix.shape[0], dtype=complex)

    # Apply Hadamard to columns i and j
    hadamard_matrix[:, i] = (1 / np.sqrt(2)) * (identity[:, i] + identity[:, j])
    hadamard_matrix[:, j] = (1 / np.sqrt(2)) * (identity[:, i] - identity[:, j])

    return hadamard_matrix @ matrix @ hadamard_matrix.conj().T


def change_basis(matrix):
    new_matrix = apply_hadamard(matrix, 6, 7)
    new_matrix = swap_basis(new_matrix, 4, 5)
    new_matrix = swap_basis(new_matrix, 5, 6)
    return new_matrix