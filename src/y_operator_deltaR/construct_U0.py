import numpy as np
from scipy.linalg import expm
from src.y_operator_deltaR.params import get_params


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
    om_0 = np.sqrt(delta ** 2 + omega ** 2)
    cos_term = np.cos(om_0 * t / 2)
    sin_term = np.sin(om_0 * t / 2)
    U_bb = cos_term - 1j * delta / om_0 * sin_term
    U_br = -1j * (omega * np.exp(1j * xi)) / om_0 * sin_term
    U_rb = -1j * (omega * np.exp(-1j * xi)) / om_0 * sin_term
    U_rr = cos_term + 1j * delta / om_0 * sin_term

    U = np.array([[U_bb, U_br], [U_rb, U_rr]], dtype=complex) * np.exp(1j * delta * t / 2)
    return U


def get_U_deltaR(delta, omega, xi, t, delta_rydberg):
    # CAREFULL WITH SQRT(2) * OMEGA
    Hamiltonian = np.zeros((3, 3), dtype=complex)
    Hamiltonian[0, 0] = 0
    Hamiltonian[0, 1] = (np.sqrt(2) * omega) / 2 * np.exp(1j * xi)
    Hamiltonian[1, 0] = Hamiltonian[0, 1].conj()
    Hamiltonian[1, 1] = - delta
    Hamiltonian[1, 2] = (np.sqrt(2) * omega) / 2 * np.exp(1j * xi)
    Hamiltonian[2, 1] = Hamiltonian[1, 2].conj()
    Hamiltonian[2, 2] = delta_rydberg - 2 * delta
    U_delta = expm(-1j * Hamiltonian * t)
    if not np.allclose(U_delta @ U_delta.conj().T, np.eye(U_delta.shape[0])):
        raise ValueError("Output matrix U_delta is not unitary!")
    return U_delta


def construct_U0(t, om, delta_rydberg):
    tau, delta, xi = get_params(om, delta_rydberg)
    if t <= tau:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, om, 0.0, t)
        U1[3:5, 3:5] = get_U(delta, om, 0.0, t)
        # U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, 0.0, t)

        U_deltaR1 = get_U_deltaR(delta, om, 0.0, t, delta_rydberg)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        U1[np.ix_(rows, cols)] = U_deltaR1
        return change_basis(U1)
    else:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, om, 0.0, tau)
        U1[3:5, 3:5] = get_U(delta, om, 0.0, tau)
        # U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, 0.0, tau)

        U_deltaR1 = get_U_deltaR(delta, om, 0.0, tau, delta_rydberg)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        U1[np.ix_(rows, cols)] = U_deltaR1

        U2 = np.eye(9, dtype=complex)
        U2[1:3, 1:3] = get_U(delta, om, xi, t - tau)
        U2[3:5, 3:5] = get_U(delta, om, xi, t - tau)
        # U2[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, xi, t - tau)

        U_deltaR2 = get_U_deltaR(delta, om, xi, t - tau, delta_rydberg)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        U2[np.ix_(rows, cols)] = U_deltaR2

        if not np.allclose((U2 @ U1) @ (U2 @ U1).conj().T, np.eye(U2.shape[0])):
            raise ValueError("Input matrix U2 is not unitary!")
        return change_basis(U2 @ U1)
