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


def get_integrand_A(params: YOperatorDerived, t, get_matrix, t_initial=0.0):
    # Measure evolution from the start of this pulse, so U0 is the identity at t_initial
    U0 = construct_U0(params, t) @ construct_U0(params, t_initial).conj().T  # for the second pulse U should start at the new initial time
    M = get_matrix(params, t)   # for functions like get_W0z, get_Wz
    if M.shape == (3, 3):
        M = np.kron(M, np.eye(3))
    return U0.conj().T @ M @ U0


def get_integrand_B(params: YOperatorDerived, t, get_matrix, t_initial=0.0):
    U0 = construct_U0(params, t) @ construct_U0(params, t_initial).conj().T
    M = get_matrix(params, t)   # for functions like get_W0z, get_Wz
    if M.shape == (3, 3):
        M = np.kron(np.eye(3), M)
    if get_matrix is get_vdW:
        M = -M
    return U0.conj().T @ M @ U0


def get_integrals_A(params: YOperatorDerived, t_initial, t_final, epsrel=1e-18):
    integrals = {}

    for key, term_list in EFF_DICT.items():
        if params.eff_dict[key]:
            for name, func_spin, _ in term_list:
                integrals[name], _ = quad_vec(
                    lambda t, f=func_spin: get_integrand_A(params, t, f, t_initial),
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
                    lambda t, f=func_spin: get_integrand_B(params, t, f, t_initial),
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
