import numpy as np
from src.algorithm.algorithm_tools import calculate_rho_wprx_wx, calculate_rho_SX
from src.algorithm.algorithm_tools import calculate_spin_density_withX


def one_iteration(rho_S0, rho_T0A, rho_T0B, U0, Y_A, Y_B, to_print=False):
    rho_SA = calculate_rho_SX(rho_S0, rho_T0A, Y_A)
    rho_SB = calculate_rho_SX(rho_S0, rho_T0B, Y_B)

    rho_S1 = calculate_spin_density_withX(rho_SB, rho_T0A, U0, Y_A)
    rho_S2 = calculate_spin_density_withX(rho_SA, rho_T0B, U0, Y_B)
    rho_S = (rho_S1 + rho_S2) / 2

    rho_TA = np.zeros((len(rho_T0A), len(rho_T0A)), dtype=complex)
    rho_TB = np.zeros((len(rho_T0B), len(rho_T0B)), dtype=complex)
    for wpr_A in range(len(rho_TA)):
        for w_A in range(len(rho_TA)):
            rho_TA[wpr_A, w_A] = np.trace(calculate_rho_wprx_wx(rho_SB, rho_T0A, Y_A, wpr_A, w_A))
    for wpr_B in range(len(rho_TB)):
        for w_B in range(len(rho_TB)):
            rho_TB[wpr_B, w_B] = np.trace(calculate_rho_wprx_wx(rho_SA, rho_T0B, Y_B, wpr_B, w_B))

    # THIS CODE WILL CHECK HOW ACCURATE THE WHOLE CALCULATION IS
    # OR HOW MUCH ENTANGLEMENT IS LOST IN THE PROCESS
    if to_print:
        fidelity_sa_sb = abs(np.trace(rho_SA @ rho_SB))
        print('\n')
        print('FIDELITY S_A S_B : ', fidelity_sa_sb)

        rho_S_alternative = calculate_spin_density_withX(rho_SA, rho_T0B, U0, Y_B)
        fidelity_spin = abs(np.trace(rho_S @ rho_S_alternative))
        print('FIDELITY SPIN : ', fidelity_spin)

        fidelity_temp = abs(np.trace(rho_TA @ rho_TB))
        print('FIDELITY TEMPERATURE : ', fidelity_temp)
        print('\n')

    return rho_S, rho_TA, rho_TB
