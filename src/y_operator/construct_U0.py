import numpy as np
from src.y_operator.params import xi2, get_params


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
    """Apply Hadamard and swap transformations to a unitary matrix."""
    if not np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0])):
        raise ValueError("Input matrix is not unitary!")

    new_matrix = apply_hadamard(matrix, 6, 7)
    new_matrix = swap_basis(new_matrix, 4, 5)
    new_matrix = swap_basis(new_matrix, 5, 6)

    # Verify unitarity after transformations
    if not np.allclose(new_matrix @ new_matrix.conj().T, np.eye(new_matrix.shape[0])):
        raise ValueError("Resulting matrix is not unitary!")

    return new_matrix


def get_U(delta, omega, xi, t):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    Ubb = (np.cos(om_0 * t / 2) + 1j * delta / om_0 * np.sin(om_0 * t / 2)) * np.exp(-1j * delta * t / 2)
    Ubr = 1j * omega / om_0 * np.sin(om_0 * t / 2) * np.exp(1j * xi - 1j * delta * t / 2)
    Urb = 1j * omega / om_0 * np.sin(om_0 * t / 2) * np.exp(-1j * xi + 1j * delta * t / 2)
    Urr = (np.cos(om_0 * t / 2) - 1j * delta / om_0 * np.sin(om_0 * t / 2)) * np.exp(1j * delta * t / 2)
    U = np.array([[Ubb, Ubr], [Urb, Urr]], dtype=complex)
    return U


def construct_U0(t, om):
    tau, delta = get_params(om)
    if t <= tau:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, om, 0.0, t)
        U1[3:5, 3:5] = get_U(delta, om, 0.0, t)
        U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, 0.0, t)
        return change_basis(U1)
    else:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, om, 0.0, tau)
        U1[3:5, 3:5] = get_U(delta, om, 0.0, tau)
        U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, 0.0, tau)

        U2 = np.eye(9, dtype=complex)
        U2[1:3, 1:3] = get_U(delta, om, xi2, t - tau)
        U2[3:5, 3:5] = get_U(delta, om, xi2, t - tau)
        U2[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, xi2, t - tau)
        return change_basis(U2 @ U1)

