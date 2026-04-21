import numpy as np
from src.algorithm.algorithm_tools import calculate_YrhoY


def one_iteration(rho_S0, rho_T0A, rho_T0B, U0, Y_A, Y_B):
    YA_rhoS0rhoT0A_YA = calculate_YrhoY(rho_S0, rho_T0A, Y_A)
    YB_rhoS0rhoT0A_YB = calculate_YrhoY(rho_S0, rho_T0B, Y_B)

    rho_SA = np.einsum('iaja->ij', YA_rhoS0rhoT0A_YA)
    rho_SB = np.einsum('iaja->ij', YB_rhoS0rhoT0A_YB)


    YA_rhoSBrhoT0A_YA = calculate_YrhoY(rho_SB, rho_T0A, Y_A)
    YB_rhoSArhoT0B_YB = calculate_YrhoY(rho_SA, rho_T0B, Y_B)

    rho_S1 = U0 @ np.einsum('iaja->ij', YA_rhoSBrhoT0A_YA) @ U0.conj().T
    rho_S2 = U0 @ np.einsum('iaja->ij', YB_rhoSArhoT0B_YB) @ U0.conj().T
    rho_S = (rho_S1 + rho_S2) / 2


    rho_TA = np.einsum('iaib->ab', YA_rhoSBrhoT0A_YA)
    rho_TB = np.einsum('iaib->ab', YB_rhoSArhoT0B_YB)

    return rho_S, rho_TA, rho_TB
