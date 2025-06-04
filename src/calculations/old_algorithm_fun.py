import numpy as np
from src.calculations.algorithm_tools import calculate_rho_w, calculate_rho_SA, calculate_rho_SB
from src.calculations.algorithm_tools import calculate_spin_density_withA, calculate_spin_density_withB


def first_iteration(rho_S0, rho_T0, U0, Y_A, Y_B):
    # Here we pretend that we knew the rho_T0 for B subsystem
    rho_SB = calculate_rho_SB(rho_S0, rho_T0, Y_B)
    rho_S = calculate_spin_density_withB(rho_SB, rho_T0, U0, Y_A)
    # now we can get rho_SA
    rho_TA = np.zeros((len(rho_T0), len(rho_T0)), dtype=complex)
    for w_A in range(len(rho_TA)):
        rho_TA[w_A, w_A] = np.trace(calculate_rho_w(rho_S, rho_T0, Y_A, w_A))
    return rho_S, rho_TA


def second_iteration(rho_S0, rho_T0, U0, Y_A, Y_B):
    rho_SA = calculate_rho_SA(rho_S0, rho_T0, Y_A)
    rho_S = calculate_spin_density_withA(rho_SA, rho_T0, U0, Y_B)

    rho_TB = np.zeros((len(rho_T0), len(rho_T0)), dtype=complex)
    for w_B in range(len(rho_T0)):
        rho_TB[w_B, w_B] = np.trace(calculate_rho_w(rho_S, rho_T0, Y_B, w_B))
    return rho_S, rho_TB


def exact_evolution(rho_S0, U0):
    return U0 @ rho_S0 @ U0.conj().T
