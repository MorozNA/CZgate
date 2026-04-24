import numpy as np
from scipy.linalg import expm
from src.y_operator.change_basis import change_basis
from src.y_operator.config import YOperatorDerived
from dataclasses import replace


def get_U(params: YOperatorDerived, t):
    om_0 = np.sqrt(params.delta ** 2 + params.om ** 2)
    cos_term = np.cos(om_0 * t / 2)
    sin_term = np.sin(om_0 * t / 2)
    U_bb = cos_term - 1j * params.delta / om_0 * sin_term
    U_br = -1j * (params.om * np.exp(1j * params.xi)) / om_0 * sin_term
    U_rb = -1j * (params.om * np.exp(-1j * params.xi)) / om_0 * sin_term
    U_rr = cos_term + 1j * params.delta / om_0 * sin_term

    U = np.array([[U_bb, U_br], [U_rb, U_rr]], dtype=complex) * np.exp(1j * params.delta * t / 2)
    return U


def get_U_deltaR(params: YOperatorDerived, t):
    H = np.zeros((3, 3), dtype=complex)
    H[0, 0] = 0
    H[0, 1] = (np.sqrt(2) * params.om) / 2 * np.exp(1j * params.xi)
    H[1, 0] = H[0, 1].conj()
    H[1, 1] = - params.delta
    H[1, 2] = (np.sqrt(2) * params.om) / 2 * np.exp(1j * params.xi)
    H[2, 1] = H[1, 2].conj()
    H[2, 2] = params.delta_rydberg - 2 * params.delta
    U_delta = expm(-1j * H * t)
    if not np.allclose(U_delta @ U_delta.conj().T, np.eye(U_delta.shape[0])):
        raise ValueError("Output matrix U_delta is not unitary!")
    return U_delta



# TODO: write comments to justify p_xi0 and other parameters or completely change the structure of the functions
def construct_U0(params: YOperatorDerived, t):
    use_deltaR = params.delta_rydberg is not None

    p_xi0_om = replace(params, xi=0.0)
    p_xi0_sqrt2om = replace(params, xi=0.0, om=np.sqrt(2)*params.om)
    p_sqrt2om = replace(params, om=np.sqrt(2)*params.om)

    if t <= params.tau:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(p_xi0_om, t)
        U1[3:5, 3:5] = get_U(p_xi0_om, t)

        if use_deltaR:
            U_deltaR1 = get_U_deltaR(p_xi0_om, t)
            rows = [5, 6, 8]  # Target rows in U1
            cols = [5, 6, 8]  # Target columns in U1
            U1[np.ix_(rows, cols)] = U_deltaR1
        else:
            U1[5:7, 5:7] = get_U(p_xi0_sqrt2om, t)

        return change_basis(U1)

    U1 = np.eye(9, dtype=complex)
    U1[1:3, 1:3] = get_U(p_xi0_om, params.tau)
    U1[3:5, 3:5] = get_U(p_xi0_om, params.tau)

    if use_deltaR:
        U_deltaR1 = get_U_deltaR(p_xi0_om, params.tau)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        U1[np.ix_(rows, cols)] = U_deltaR1
    else:
        U1[5:7, 5:7] = get_U(p_xi0_sqrt2om, params.tau)



    U2 = np.eye(9, dtype=complex)
    U2[1:3, 1:3] = get_U(params, t - params.tau)
    U2[3:5, 3:5] = get_U(params, t - params.tau)

    if use_deltaR:
        U_deltaR2 = get_U_deltaR(params, t - params.tau)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        U2[np.ix_(rows, cols)] = U_deltaR2
    else:
        U2[5:7, 5:7] = get_U(p_sqrt2om, t - params.tau)

    if not np.allclose((U2 @ U1) @ (U2 @ U1).conj().T, np.eye(U2.shape[0])):
        raise ValueError("Input matrix U2 is not unitary!")
    return change_basis(U2 @ U1)
