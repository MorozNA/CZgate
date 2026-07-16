import numpy as np
from src.y_operator.config import YOperatorDerived
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.internal import get_V1, get_V2, get_W0z, get_Wz, get_vdW
from src.y_operator.motional import get_V1_mot, get_V2_mot, get_W0z_mot, get_Wz_mot, get_vdW_mot
from scipy.integrate import quad_vec
from tqdm import tqdm


EFF_DICT = {
    "always": [
        ("V1", get_V1, get_V1_mot)
    ],
    "Q": [
        ("V2", get_V2, get_V2_mot)
    ],
    "W": [
        ("W0z", get_W0z, get_W0z_mot),
        ("Wz", get_Wz, get_Wz_mot),
    ],
    "vdW": [
        ("vdW", get_vdW, get_vdW_mot),
    ],
}


def get_integrand_A(params: YOperatorDerived, t, get_matrix):
    U0 = construct_U0(params, t)
    M = get_matrix(params, t)
    if len(M)==9:
        return U0 @ M @ U0.conj().T
    return U0 @ np.kron(M, np.eye(3)) @ U0.conj().T


def get_integrand_B(params: YOperatorDerived, t, get_matrix):
    U0 = construct_U0(params, t)
    M = get_matrix(params, t)   # for functions like get_W0z, get_Wz
    if len(M)==9:
        # TODO: include the minus sign in other way
        return -U0 @ M @ U0.conj().T
    return U0 @ np.kron(np.eye(3), M) @ U0.conj().T


def get_integrals_A(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    integrals = {}

    for key, term_list in EFF_DICT.items():
        if params.eff_dict[key]:
            for name, func_spin, _ in term_list:
                integrals[name], _ = quad_vec(
                    lambda t, f=func_spin: get_integrand_A(params, t, f),
                    t_initial,
                    t_final,
                    epsrel=epsrel,
                )
    return integrals


def get_integrals_B(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    integrals = {}

    for key, term_list in EFF_DICT.items():
        if params.eff_dict[key]:
            for name, func_spin, _ in term_list:
                integrals[name], _ = quad_vec(
                    lambda t, f=func_spin: get_integrand_B(params, t, f),
                    t_initial,
                    t_final,
                    epsrel=epsrel,
                )
    return integrals


def get_integral_atom_A(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    int_atomA = 0

    integrals = get_integrals_A(params, t_initial, t_final, epsrel)

    for key, term_list in EFF_DICT.items():
        if params.eff_dict[key]:
            for name, _, func_mot in term_list:
                int_atomA += np.kron(integrals[name], func_mot(params))

    return int_atomA


def get_integral_atom_B(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    int_atomB = 0

    integrals = get_integrals_B(params, t_initial, t_final, epsrel)

    for key, term_list in EFF_DICT.items():
        if params.eff_dict[key]:
            for name, _, func_mot in term_list:
                int_atomB += np.kron(integrals[name], func_mot(params))
    return int_atomB
