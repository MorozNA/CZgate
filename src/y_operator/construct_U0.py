import numpy as np
from scipy.linalg import expm
from src.y_operator.change_basis import change_basis


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
    H = np.zeros((3, 3), dtype=complex)
    H[0, 0] = 0
    H[0, 1] = (np.sqrt(2) * omega) / 2 * np.exp(1j * xi)
    H[1, 0] = H[0, 1].conj()
    H[1, 1] = - delta
    H[1, 2] = (np.sqrt(2) * omega) / 2 * np.exp(1j * xi)
    H[2, 1] = H[1, 2].conj()
    H[2, 2] = delta_rydberg - 2 * delta
    U_delta = expm(-1j * H * t)
    if not np.allclose(U_delta @ U_delta.conj().T, np.eye(U_delta.shape[0])):
        raise ValueError("Output matrix U_delta is not unitary!")
    return U_delta


def construct_U0(t, om, tau, delta, xi, delta_rydberg=None):
    use_deltaR = delta_rydberg is not None

    if t <= tau:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, om, 0.0, t)
        U1[3:5, 3:5] = get_U(delta, om, 0.0, t)

        if use_deltaR:
            U_deltaR1 = get_U_deltaR(delta, om, 0.0, t, delta_rydberg)
            rows = [5, 6, 8]  # Target rows in U1
            cols = [5, 6, 8]  # Target columns in U1
            U1[np.ix_(rows, cols)] = U_deltaR1
        else:
            U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, 0.0, t)

        return change_basis(U1)

    U1 = np.eye(9, dtype=complex)
    U1[1:3, 1:3] = get_U(delta, om, 0.0, tau)
    U1[3:5, 3:5] = get_U(delta, om, 0.0, tau)

    if use_deltaR:
        U_deltaR1 = get_U_deltaR(delta, om, 0.0, tau, delta_rydberg)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        U1[np.ix_(rows, cols)] = U_deltaR1
    else:
        U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, 0.0, tau)



    U2 = np.eye(9, dtype=complex)
    U2[1:3, 1:3] = get_U(delta, om, xi, t - tau)
    U2[3:5, 3:5] = get_U(delta, om, xi, t - tau)

    if use_deltaR:
        U_deltaR2 = get_U_deltaR(delta, om, xi, t - tau, delta_rydberg)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        U2[np.ix_(rows, cols)] = U_deltaR2
    else:
        U2[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, xi, t - tau)

    if not np.allclose((U2 @ U1) @ (U2 @ U1).conj().T, np.eye(U2.shape[0])):
        raise ValueError("Input matrix U2 is not unitary!")
    return change_basis(U2 @ U1)
