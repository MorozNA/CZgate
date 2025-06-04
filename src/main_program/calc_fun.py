import numpy as np
from initial_state import get_rho_T0
from evolution_fun import get_rho_T_partial, get_rho_S_partial
from evolution_fun import get_rho_s
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator.params import get_params


def get_rho_expected(rho_S0, omega):
    tau, delta = get_params(omega)
    U0 = construct_U0(2 * tau, omega)
    return U0 @ rho_S0 @ U0.conj().T


def get_rho_t(rho_S0, omega, q, temperature):
    tau, _ = get_params(omega)
    Y_A = construct_Y_A(0.0, 2 * tau, omega, q)
    Y_B = construct_Y_B(0.0, 2 * tau, omega, q)
    U0 = construct_U0(2 * tau, omega)

    rho_T0 = get_rho_T0(temperature)

    rho_S_part_B = get_rho_S_partial(rho_S0, rho_T0, Y_B)
    rho_S = get_rho_s(rho_S_part_B, rho_T0, Y_A, U0)
    return rho_S / np.trace(rho_S)


from src.y_operator.params import HBAR, OM_small, M


def get_fidelity_expected(tau, q):
    p = np.sqrt(HBAR * OM_small * M / 2) * 1
    x0 = 1.0e-6
    alpha = (HBAR * q + p) / M * 2 * tau / x0
    return 1 / 2 * (1 + np.exp(-2 * abs(alpha) ** 2 / 2))