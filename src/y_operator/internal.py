import numpy as np
from src.y_operator.constants import HBAR, M
from src.y_operator.config import YOperatorDerived


def get_OM_eff(params: YOperatorDerived, t):
    om_eff = np.sqrt(2 * params.om ** 2 + params.delta ** 2)
    if t <= params.tau:
        return om_eff
    else:
        return om_eff * np.exp(1j * params.xi)


def get_V1(params: YOperatorDerived):
    const_V1_spin = - 1.0 / HBAR / 2.0 / M
    const_V1_vib = HBAR * params.OM_small * M / 2
    const_V1 = const_V1_vib * const_V1_spin
    V_matrix1 = np.eye(3, dtype=complex) * const_V1
    return V_matrix1


def get_V2(params: YOperatorDerived):
    const_V2_spin = - 1.0 / HBAR / 2.0 / M
    const_V2_vib = params.p0z ** 2  # characteristic value of momentum, V2_vib is normalized by this value
    const_V2 = const_V2_spin * const_V2_vib
    V_matrix2 = np.zeros((3, 3), dtype=complex)
    V_matrix2[2, 2] = const_V2
    return V_matrix2


def get_W0z(params: YOperatorDerived, t):
    const_w0z = params.W_INT_CONSTANT * get_OM_eff(params, t).conj() * 1j * HBAR / params.Z_ast / params.p0z
    W0z_matrix = np.zeros((3, 3), dtype=complex)
    W0z_matrix[1, 2] = const_w0z
    W0z_matrix[2, 1] = const_w0z.conj()
    return W0z_matrix


def get_Wz(params: YOperatorDerived, t):
    const_wz = params.W_INT_CONSTANT * (HBAR ** 2) / (params.p0z ** 2)
    Wz_matrix = np.zeros((3, 3), dtype=complex)
    Wz_matrix[0, 0] = -params.DELTA_a
    Wz_matrix[1, 1] = -params.DELTA_b
    Wz_matrix[1, 2] = get_OM_eff(params, t).conj() / 2
    Wz_matrix[2, 1] = get_OM_eff(params, t) / 2
    Wz_matrix[2, 2] = -params.DELTA_r
    return const_wz * params.z_ij_matrix * Wz_matrix

