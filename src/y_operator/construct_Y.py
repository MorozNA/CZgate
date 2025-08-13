import numpy as np
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.consctust_spin_matrices import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator.construct_vib_matrices import get_V1_vib, get_V2_vib, get_W0z_vib, get_Wz_vib
from scipy.linalg import expm
from scipy.integrate import quad_vec


def get_integrand_A(t, om, matrix_fun):
    U0 = construct_U0(t, om)
    integrand = U0 @ (np.kron(matrix_fun(t, om), np.eye(3))) @ U0.conj().T
    return integrand


def get_integrand_B(t, om, matrix_fun):
    U0 = construct_U0(t, om)
    integrand = U0 @ (np.kron(np.eye(3), matrix_fun(t, om))) @ U0.conj().T
    return integrand


def integrate_matrix_A(t_initial, t_final, om, matrix_fun):
    res, error = quad_vec(lambda t: get_integrand_A(t, om, matrix_fun), t_initial, t_final)
    return res


def integrate_matrix_B(t_initial, t_final, om, matrix_fun):
    res, error = quad_vec(lambda t: get_integrand_B(t, om, matrix_fun), t_initial, t_final)
    return res


def get_integral_atom_A(t_initial, t_final, om, q, n):
    integral_V1 = integrate_matrix_A(t_initial, t_final, om, get_V1)
    integral_V2 = integrate_matrix_A(t_initial, t_final, om, get_V2)
    integral_W0z = integrate_matrix_A(t_initial, t_final, om, get_W0z)
    integral_Wz = integrate_matrix_A(t_initial, t_final, om, get_Wz)

    integral_V1 = np.kron(integral_V1, get_V1_vib(n))
    integral_V2 = np.kron(integral_V2, get_V2_vib(n, q))
    integral_W0z = np.kron(integral_W0z, get_W0z_vib(n))
    integral_Wz = np.kron(integral_Wz, get_Wz_vib(n))

    integral_atomA = integral_V1 + integral_V2 + integral_W0z + integral_Wz
    return integral_atomA


def construct_Y_A(t_initial, t_final, om, q, n):
    integral_atomA = get_integral_atom_A(t_initial, t_final, om, q, n)
    Y_A = expm(-1j * integral_atomA)  # division by HBAR is already accounted for in matrix definition
    return Y_A


def get_integral_atom_B(t_initial, t_final, om, q, n):
    integral_V1 = integrate_matrix_B(t_initial, t_final, om, get_V1)
    integral_V2 = integrate_matrix_B(t_initial, t_final, om, get_V2)
    integral_W0z = integrate_matrix_B(t_initial, t_final, om, get_W0z)
    integral_Wz = integrate_matrix_B(t_initial, t_final, om, get_Wz)

    integral_V1 = np.kron(integral_V1, get_V1_vib(n))
    integral_V2 = np.kron(integral_V2, get_V2_vib(n, q))
    integral_W0z = np.kron(integral_W0z, get_W0z_vib(n))
    integral_Wz = np.kron(integral_Wz, get_Wz_vib(n))

    integral_atomB = integral_V1 + integral_V2 + integral_W0z + integral_Wz
    return integral_atomB


def construct_Y_B(t_initial, t_final, om, q, n):
    integral_atomB = get_integral_atom_B(t_initial, t_final, om, q, n)
    Y_B = expm(-1j * integral_atomB)  # division by HBAR is already accounted for in matrix definition
    return Y_B
