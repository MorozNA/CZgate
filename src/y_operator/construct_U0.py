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


def construct_U0k(params: YOperatorDerived, t):
    use_deltaR = params.delta_rydberg is not None
    Uk = np.eye(9, dtype=complex)
    Uk[1:3, 1:3] = get_U(params, t)
    Uk[3:5, 3:5] = get_U(params, t)

    if use_deltaR:
        U_deltaR1 = get_U_deltaR(params, t)
        rows = [5, 6, 8]
        cols = [5, 6, 8]
        Uk[np.ix_(rows, cols)] = U_deltaR1
    else:
        Uk[5:7, 5:7] = get_U(replace(params, om=np.sqrt(2)*params.om), t)
    return change_basis(Uk)


def construct_U0(params: YOperatorDerived, t):
    if t <= params.tau:
        return construct_U0k(params, t)
    U01 = construct_U0k(replace(params, xi=0.0), t)
    U02 = construct_U0k(params, t - params.tau)
    return U02 @ U01
