import numpy as np
from src.y_operator.params import get_params
from src.y_operator.params import xi2, HBAR, M, p0z, Z_ast
from src.y_operator.params import z_ij_matrix, DELTA_a, DELTA_b, DELTA_r
from src.y_operator.params import W_INT_CONSTANT
from src.y_operator.params import OM_small


def get_xi(t, om):
    tau, _ = get_params(om)
    if t <= tau:
        return 0.0
    else:
        return xi2


def get_OM_eff(t, om):
    _, delta = get_params(om)
    om_eff = np.sqrt(2 * om ** 2 + delta ** 2)
    return om_eff * np.exp(1j * get_xi(t, om))


# Constructing both parts of V_matrix
const_V1_spin = - 1.0 / HBAR / 2.0 / M
const_V1_vib = HBAR * OM_small * M / 2
const_V1 = const_V1_vib * const_V1_spin
V_matrix1 = np.eye(3, dtype=complex) * const_V1


def get_V1(t, om):
    return V_matrix1


const_V2_spin = - 1.0 / HBAR / 2.0 / M
const_V2_vib = p0z ** 2  # characteristic value of momentum, V2_vib is normalized by this value
const_V2 = const_V1_spin * const_V2_vib
V_matrix2 = np.zeros((3, 3), dtype=complex)
V_matrix2[2, 2] = const_V2


def get_V2(t, om):
    return V_matrix2


def get_W0z(t, om):
    const_w0z = W_INT_CONSTANT * get_OM_eff(t, om).conj() * 1j * HBAR / Z_ast / p0z
    W0z_matrix = np.zeros((3, 3), dtype=complex)
    W0z_matrix[1, 2] = const_w0z
    W0z_matrix[2, 1] = const_w0z.conj()
    return W0z_matrix


def get_Wz(t, om):
    const_wz = W_INT_CONSTANT * (HBAR ** 2) / (p0z ** 2)
    Wz_matrix = np.zeros((3, 3), dtype=complex)
    Wz_matrix[0, 0] = -DELTA_a
    Wz_matrix[1, 1] = -DELTA_b
    Wz_matrix[1, 2] = get_OM_eff(t, om).conj() / 2
    Wz_matrix[2, 1] = get_OM_eff(t, om) / 2
    Wz_matrix[2, 2] = -DELTA_r
    return const_wz * z_ij_matrix * Wz_matrix

