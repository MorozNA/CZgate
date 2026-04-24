import numpy as np
from src.y_operator.constants import HBAR
from src.y_operator.config import YOperatorDerived


def get_V1_mot(params: YOperatorDerived):
    # note that p squared is already normalized by p0z^2
    nu = np.arange(params.n)
    p2_1 = np.diag(2 * nu + 1)
    p2_2 = np.diag(np.sqrt(nu[2::] * (nu[2::] - 1)), 2)
    p2_3 = np.diag(np.sqrt(nu[2::] * (nu[2::] - 1)), -2)
    return p2_1 - p2_2 - p2_3


def get_V2_mot(params: YOperatorDerived):
    # note that p1 is already normalized by p0z
    nu = np.arange(params.n)
    p1_1 = np.diag(np.sqrt(nu[1::]), 1)
    p1_2 = np.diag(np.sqrt(nu[1::]), -1)
    p1 = (p1_1 - p1_2) * (-1j)
    term1 = 2 * (HBAR * (params.q * params.Q_INT_CONSTANT) / params.p0z) * p1
    term2 = (HBAR * (params.q * params.Q_INT_CONSTANT) / params.p0z) ** 2 * np.eye(params.n, dtype=complex)
    return term1 + term2


def get_W0z_mot(params: YOperatorDerived):
    nu = np.arange(params.n)
    p1_1 = np.diag(np.sqrt(nu[1::]), 1)
    p1_2 = np.diag(np.sqrt(nu[1::]), -1)
    return p1_1 + p1_2


def get_Wz_mot(params: YOperatorDerived):
    nu = np.arange(params.n)
    p2_1 = np.diag(2 * nu + 1)
    p2_2 = np.diag(np.sqrt(nu[2::] * (nu[2::] - 1)), 2)
    p2_3 = np.diag(np.sqrt(nu[2::] * (nu[2::] - 1)), -2)
    return -(p2_1 + p2_2 + p2_3)
