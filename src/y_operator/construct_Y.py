import numpy as np
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.consctust_spin_matrices import get_A, get_V1, get_V2, get_W0z
from src.y_operator.construct_vib_matrices import A_vib, V1_vib, V2_vib, W0z_vib, n
from scipy.linalg import expm
from scipy.integrate import quad_vec


# TODO: Логичнее будет генерировать U0(t) один раз
def get_integrand(t, matrix_fun):
    U0 = construct_U0(t)
    integrand = U0 @ (np.kron(matrix_fun(t), np.eye(3))) @ U0.conj().T # TODO: make order changable to get Y_B
    integrand = 0.5 * (integrand + integrand.conj().T)  # TODO: get rid of this
    return integrand


# TODO: add completion bar
def integrate_matrix(t_initial, t_final, matrix_fun):
    res, error = quad_vec(lambda t: get_integrand(t, matrix_fun), t_initial, t_final)
    return res


def construct_Y_A(t_initial, t_final):
    integral_A = integrate_matrix(t_initial, t_final, get_A)
    integral_V1 = integrate_matrix(t_initial, t_final, get_V1)
    integral_V2 = integrate_matrix(t_initial, t_final, get_V2)
    integral_W0z = integrate_matrix(t_initial, t_final, get_W0z)

    # integral_A = np.einsum('ij,ab->ijab', integral_A, A_vib) doesn't work smh
    integral_A = np.einsum('ij,ab->iajb', integral_A, A_vib)
    integral_V1 = np.einsum('ij,ab->iajb', integral_V1, V1_vib)
    integral_V2 = np.einsum('ij,ab->iajb', integral_V2, V2_vib)
    integral_W0z = np.einsum('ij,ab->iajb', integral_W0z, W0z_vib)

    integral_atomA = integral_A + integral_V1 + integral_V2 + integral_W0z
    # print('integral_atomA = ', np.amax(abs(integral_atomA)))

    Y_A = expm(-1j * integral_atomA.reshape(9 * n, 9 * n))  # division by HBAR is already accounted for in matrix def
    return Y_A


from scipy.sparse import kron, identity, csr_matrix


def construct_Y(t_initial, t_final):
    Y = construct_Y_A(t_initial, t_final).reshape(9, n, 9, n)
    I_n = identity(n, format='csr')
    Y_A = kron(csr_matrix(Y.transpose([1, 0, 3, 2]).reshape(9 * n, 9 * n)), I_n)
    Y_B = kron(I_n, csr_matrix(Y.reshape(9 * n, 9 * n)))
    return Y_A, Y_B
