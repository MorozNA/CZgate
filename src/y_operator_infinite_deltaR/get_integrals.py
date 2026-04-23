import numpy as np
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.consctust_spin_matrices import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator.construct_vib_matrices import get_V1_vib, get_V2_vib, get_W0z_vib, get_Wz_vib
from scipy.integrate import quad_vec


def get_integrand_V1_A(t, om, tau, delta, xi):
    U0 = construct_U0(t, om, tau, delta, xi)
    integrand_V1 = U0 @ (np.kron(get_V1(), np.eye(3))) @ U0.conj().T
    return integrand_V1


def get_integrand_V2_A(t, om, tau, delta, xi):
    U0 = construct_U0(t, om, tau, delta, xi)
    integrand_V2 = U0 @ (np.kron(get_V2(), np.eye(3))) @ U0.conj().T
    return integrand_V2


def get_integrand_W0z_A(t, om, tau, delta, xi):
    U0 = construct_U0(t, om, tau, delta, xi)
    integrand_W0z = U0 @ (np.kron(get_W0z(t, om, tau, delta, xi), np.eye(3))) @ U0.conj().T
    return integrand_W0z


def get_integrand_Wz_A(t, om, tau, delta, xi):
    U0 = construct_U0(t, om, tau, delta, xi)
    integrand_Wz = U0 @ (np.kron(get_Wz(t, om, tau, delta, xi), np.eye(3))) @ U0.conj().T
    return integrand_Wz


def get_integrand_V1_B(t, om, tau, delta, xi):
    U0 = construct_U0(t, om, tau, delta, xi)
    integrand_V1 = U0 @ (np.kron(np.eye(3), get_V1())) @ U0.conj().T
    return integrand_V1


def get_integrand_V2_B(t, om, tau, delta, xi):
    U0 = construct_U0(t, om, tau, delta, xi)
    integrand_V2 = U0 @ (np.kron(np.eye(3), get_V2())) @ U0.conj().T
    return integrand_V2


def get_integrand_W0z_B(t, om, tau, delta, xi):
    U0 = construct_U0(t, om, tau, delta, xi)
    integrand_W0z = U0 @ (np.kron(np.eye(3), get_W0z(t, om, tau, delta, xi))) @ U0.conj().T
    return integrand_W0z


def get_integrand_Wz_B(t, om, tau, delta, xi):
    U0 = construct_U0(t, om, tau, delta, xi)
    integrand_Wz = U0 @ (np.kron(np.eye(3), get_Wz(t, om, tau, delta, xi))) @ U0.conj().T
    return integrand_Wz


def get_integrals_A(t_initial, t_final, om, tau, delta, xi):
    integral_V1, error = quad_vec(lambda t: get_integrand_V1_A(t, om, tau, delta, xi), t_initial, t_final)
    integral_V2, error = quad_vec(lambda t: get_integrand_V2_A(t, om, tau, delta, xi), t_initial, t_final)
    integral_W0z, error = quad_vec(lambda t: get_integrand_W0z_A(t, om, tau, delta, xi), t_initial, t_final)
    integral_Wz, error = quad_vec(lambda t: get_integrand_Wz_A(t, om, tau, delta, xi), t_initial, t_final)
    return integral_V1, integral_V2, integral_W0z, integral_Wz


def get_integrals_B(t_initial, t_final, om, tau, delta, xi):
    integral_V1, error = quad_vec(lambda t: get_integrand_V1_B(t, om, tau, delta, xi), t_initial, t_final)
    integral_V2, error = quad_vec(lambda t: get_integrand_V2_B(t, om, tau, delta, xi), t_initial, t_final)
    integral_W0z, error = quad_vec(lambda t: get_integrand_W0z_B(t, om, tau, delta, xi), t_initial, t_final)
    integral_Wz, error = quad_vec(lambda t: get_integrand_Wz_B(t, om, tau, delta, xi), t_initial, t_final)
    return integral_V1, integral_V2, integral_W0z, integral_Wz


def get_integral_atom_A(t_initial, t_final, om, tau, delta, xi, q, n):
    integral_V1, integral_V2, integral_W0z, integral_Wz = get_integrals_A(t_initial, t_final, om, tau, delta, xi)

    integral_V1 = np.kron(integral_V1, get_V1_vib(n))
    integral_V2 = np.kron(integral_V2, get_V2_vib(n, q))
    integral_W0z = np.kron(integral_W0z, get_W0z_vib(n))
    integral_Wz = np.kron(integral_Wz, get_Wz_vib(n))

    integral_atomA = integral_V1 + integral_V2 + integral_W0z + integral_Wz
    return integral_atomA


def get_integral_atom_B(t_initial, t_final, om, tau, delta, xi, q, n):
    integral_V1, integral_V2, integral_W0z, integral_Wz = get_integrals_B(t_initial, t_final, om, tau, delta, xi)

    integral_V1 = np.kron(integral_V1, get_V1_vib(n))
    integral_V2 = np.kron(integral_V2, get_V2_vib(n, q))
    integral_W0z = np.kron(integral_W0z, get_W0z_vib(n))
    integral_Wz = np.kron(integral_Wz, get_Wz_vib(n))

    integral_atomB = integral_V1 + integral_V2 + integral_W0z + integral_Wz
    return integral_atomB