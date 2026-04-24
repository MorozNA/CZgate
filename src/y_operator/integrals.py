import numpy as np
from src.y_operator.config import YOperatorDerived
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.internal import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator.motional import get_V1_mot, get_V2_mot, get_W0z_mot, get_Wz_mot
from scipy.integrate import quad_vec


def get_integrand_V1_A(params: YOperatorDerived, t):
    U0 = construct_U0(params, t)
    integrand_V1 = U0 @ (np.kron(get_V1(params), np.eye(3))) @ U0.conj().T
    return integrand_V1


def get_integrand_V2_A(params: YOperatorDerived, t):
    U0 = construct_U0(params, t)
    integrand_V2 = U0 @ (np.kron(get_V2(params), np.eye(3))) @ U0.conj().T
    return integrand_V2


def get_integrand_W0z_A(params: YOperatorDerived, t):
    U0 = construct_U0(params, t)
    integrand_W0z = U0 @ (np.kron(get_W0z(params, t), np.eye(3))) @ U0.conj().T
    return integrand_W0z


def get_integrand_Wz_A(params: YOperatorDerived, t):
    U0 = construct_U0(params, t)
    integrand_Wz = U0 @ (np.kron(get_Wz(params, t), np.eye(3))) @ U0.conj().T
    return integrand_Wz


def get_integrand_V1_B(params: YOperatorDerived, t):
    U0 = construct_U0(params, t)
    integrand_V1 = U0 @ (np.kron(np.eye(3), get_V1(params))) @ U0.conj().T
    return integrand_V1


def get_integrand_V2_B(params: YOperatorDerived, t):
    U0 = construct_U0(params, t)
    integrand_V2 = U0 @ (np.kron(np.eye(3), get_V2(params))) @ U0.conj().T
    return integrand_V2


def get_integrand_W0z_B(params: YOperatorDerived, t):
    U0 = construct_U0(params, t)
    integrand_W0z = U0 @ (np.kron(np.eye(3), get_W0z(params, t))) @ U0.conj().T
    return integrand_W0z


def get_integrand_Wz_B(params: YOperatorDerived, t):
    U0 = construct_U0(params, t)
    integrand_Wz = U0 @ (np.kron(np.eye(3), get_Wz(params, t))) @ U0.conj().T
    return integrand_Wz


def get_integrals_A(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    integral_V1, error = quad_vec(lambda t: get_integrand_V1_A(params, t), t_initial, t_final, epsrel=epsrel)
    integral_V2, error = quad_vec(lambda t: get_integrand_V2_A(params, t), t_initial, t_final, epsrel=epsrel)
    integral_W0z, error = quad_vec(lambda t: get_integrand_W0z_A(params, t), t_initial, t_final, epsrel=epsrel)
    integral_Wz, error = quad_vec(lambda t: get_integrand_Wz_A(params, t), t_initial, t_final, epsrel=epsrel)
    return integral_V1, integral_V2, integral_W0z, integral_Wz


def get_integrals_B(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    integral_V1, error = quad_vec(lambda t: get_integrand_V1_B(params, t), t_initial, t_final, epsrel=epsrel)
    integral_V2, error = quad_vec(lambda t: get_integrand_V2_B(params, t), t_initial, t_final, epsrel=epsrel)
    integral_W0z, error = quad_vec(lambda t: get_integrand_W0z_B(params, t), t_initial, t_final, epsrel=epsrel)
    integral_Wz, error = quad_vec(lambda t: get_integrand_Wz_B(params, t), t_initial, t_final, epsrel=epsrel)
    return integral_V1, integral_V2, integral_W0z, integral_Wz


def get_integral_atom_A(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    int_V1, int_V2, int_W0z, int_Wz = get_integrals_A(params, t_initial, t_final, epsrel)

    int_V1 = np.kron(int_V1, get_V1_mot(params))
    int_V2 = np.kron(int_V2, get_V2_mot(params))
    int_W0z = np.kron(int_W0z, get_W0z_mot(params))
    int_Wz = np.kron(int_Wz, get_Wz_mot(params))

    int_atomA = int_V1 + int_V2 + int_W0z + int_Wz
    return int_atomA


def get_integral_atom_B(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    int_V1, int_V2, int_W0z, int_Wz = get_integrals_B(params, t_initial, t_final, epsrel)

    int_V1 = np.kron(int_V1, get_V1_mot(params))
    int_V2 = np.kron(int_V2, get_V2_mot(params))
    int_W0z = np.kron(int_W0z, get_W0z_mot(params))
    int_Wz = np.kron(int_Wz, get_Wz_mot(params))

    int_atomB = int_V1 + int_V2 + int_W0z + int_Wz
    return int_atomB