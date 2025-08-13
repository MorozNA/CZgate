import numpy as np
from src.y_operator_deltaR.algorithm.algorithm_tools import calculate_rho_wx, calculate_rho_SX
from src.y_operator_deltaR.algorithm.algorithm_tools import calculate_spin_density_withX


def one_iteration(rho_S0, rho_T0A, rho_T0B, U0, Y_A, Y_B):
    rho_SA = calculate_rho_SX(rho_S0, rho_T0A, Y_A)
    rho_SB = calculate_rho_SX(rho_S0, rho_T0B, Y_B)

    rho_S1 = calculate_spin_density_withX(rho_SB, rho_T0A, U0, Y_A)
    rho_S2 = calculate_spin_density_withX(rho_SA, rho_T0B, U0, Y_B)
    rho_S = (rho_S1 + rho_S2) / 2

    rho_TA = np.zeros((len(rho_T0A), len(rho_T0A)), dtype=complex)
    rho_TB = np.zeros((len(rho_T0B), len(rho_T0B)), dtype=complex)
    for w_A in range(len(rho_TA)):
        rho_TA[w_A, w_A] = np.trace(calculate_rho_wx(rho_SB, rho_T0A, Y_A, w_A))
    for w_B in range(len(rho_T0B)):
        rho_TB[w_B, w_B] = np.trace(calculate_rho_wx(rho_SA, rho_T0B, Y_B, w_B))

    return rho_S, rho_TA, rho_TB


def exact_evolution(rho_S0, U0):
    return U0 @ rho_S0 @ U0.conj().T
