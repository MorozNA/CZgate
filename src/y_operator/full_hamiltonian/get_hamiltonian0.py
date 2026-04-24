import numpy as np
from src.y_operator.change_basis import change_basis
from src.y_operator.config import YOperatorDerived
from dataclasses import replace


def get_H_simple(params: YOperatorDerived):
    H = np.zeros((2, 2), dtype=complex)
    H[0, 0] = 0.0
    H[0, 1] = params.om / 2 * np.exp(1j * params.xi)
    H[1, 0] = H[0, 1].conj()
    H[1, 1] = -params.delta
    return H


def get_H_deltaR(params: YOperatorDerived):
    H = np.zeros((3, 3), dtype=complex)
    H[0, 0] = 0
    H[0, 1] = (np.sqrt(2) * params.om) / 2 * np.exp(1j * params.xi)
    H[1, 0] = H[0, 1].conj()
    H[1, 1] = - params.delta
    H[1, 2] = (np.sqrt(2) * params.om) / 2 * np.exp(1j * params.xi)
    H[2, 1] = H[1, 2].conj()
    H[2, 2] = params.delta_rydberg - 2 * params.delta
    return H


def get_H0(params: YOperatorDerived):
    use_deltaR = params.delta_rydberg is not None

    H = np.zeros((9, 9), dtype=complex)
    H[1:3, 1:3] = get_H_simple(params)
    H[3:5, 3:5] = get_H_simple(params)

    if use_deltaR:
        H_deltaR1 = get_H_deltaR(params)
        rows = [5, 6, 8]  # Target rows in U1
        cols = [5, 6, 8]  # Target columns in U1
        H[np.ix_(rows, cols)] = H_deltaR1
    else:
        p_sqrt2om = replace(params, om=np.sqrt(2)*params.om)
        H[5:7, 5:7] = get_H_simple(p_sqrt2om)

    return change_basis(H)
