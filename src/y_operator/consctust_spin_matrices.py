import numpy as np
from src.y_operator.params import OM, xi2, tau, HBAR, M, delta, p0z, Z_ast
from src.y_operator.params import A_INT_CONSTANT, V_INT_CONSTANT, W_INT_CONSTANT


def get_xi(t):
    if t <= tau:
        return 0.0
    else:
        return xi2


def get_OM_eff(t):
    om_eff = np.sqrt(2 * OM ** 2 + delta ** 2)  # TODO: this is from presentation and should be checked
    return om_eff * np.exp(1j * get_xi(t))


def get_A(t):
    const_A01 = get_OM_eff(t) / 2 * A_INT_CONSTANT
    A_matrix = np.zeros((3, 3), dtype=complex)
    A_matrix[2, 1] = const_A01
    A_matrix[1, 2] = A_matrix[2, 1].conj()
    A_matrix[2, 2] = delta * A_INT_CONSTANT
    return A_matrix


# Also V_matrix does not depend on time, it is still convinient to use method for it
def get_V1(t):
    const_V1 = - 1.0 / HBAR / 2.0 / M * V_INT_CONSTANT
    V_matrix1 = np.zeros((3, 3), dtype=complex)
    V_matrix1[0, 0] = const_V1  # p^2 / 2m, applied to state |a>
    V_matrix1[1, 1] = const_V1
    return V_matrix1


def get_V2(t):
    const_V2 = - 1.0 / HBAR / 2.0 / M * V_INT_CONSTANT
    V_matrix2 = np.zeros((3, 3), dtype=complex)
    V_matrix2[2, 2] = const_V2
    return V_matrix2


def get_W0z(t):
    const_w0z = W_INT_CONSTANT * 1j * get_OM_eff(t) * HBAR / Z_ast / p0z  # TODO: check, whether HBAR should be here
    W0z_matrix = np.zeros((3, 3), dtype=complex)
    W0z_matrix[1, 2] = const_w0z
    W0z_matrix[2, 1] = W0z_matrix[0, 1].conj()
    return W0z_matrix
