import numpy as np
from src.calculations.algorithm_fun import one_iteration, exact_evolution
from src.calculations.initial_parameters import rho_S0
from src.calculations.initial_parameters import rho_TA_0, rho_TA_1, rho_TA_2, rho_TA_3
from src.calculations.initial_parameters import rho_TB_0, rho_TB_1, rho_TB_2, rho_TB_3
from src.calculations.initial_parameters import U0, Y_A, Y_B
from src.calculations.algorithm_tools import calculate_rho_wx, calculate_rho_SX, calculate_spin_density_withX


rho_S0_0 = np.copy(rho_S0)
rho_S0_1 = np.copy(rho_S0)
rho_S0_2 = np.copy(rho_S0)
rho_S0_3 = np.copy(rho_S0)

# calc_rho_0, rho_TA_0, rho_TB_0 = one_iteration(rho_S0_0, rho_TA_0, rho_TB_0, U0, Y_A, Y_B)
# calc_rho_1, rho_TA_1, rho_TB_1 = one_iteration(rho_S0_1, rho_TA_1, rho_TB_1, U0, Y_A, Y_B)
# calc_rho_2, rho_TA_2, rho_TB_2 = one_iteration(rho_S0_2, rho_TA_2, rho_TB_2, U0, Y_A, Y_B)
# calc_rho_3, rho_TA_3, rho_TB_3 = one_iteration(rho_S0_3, rho_TA_3, rho_TB_3, U0, Y_A, Y_B)


def test_traces():
    calc_rhos = [calc_rho_0, calc_rho_1, calc_rho_2, calc_rho_3]
    rhos_TA = [rho_TA_0, rho_TA_1, rho_TA_2, rho_TA_3]
    rhos_TB = [rho_TB_0, rho_TB_1, rho_TB_2, rho_TB_3]

    all_rhos = calc_rhos + rhos_TA + rhos_TB

    for rho in all_rhos:
        assert np.allclose(abs(np.trace(rho)), 1.0, atol=1e-10), \
            f"Trace for {rho.__name__} is not equal to 1.0"


def test_spin_matrix():
    rho_SA = calculate_rho_SX(rho_S0, rho_TA_3, Y_A)
    rho_SB = calculate_rho_SX(rho_S0, rho_TB_3, Y_B)

    rho_S_withA = calculate_spin_density_withX(rho_SA, rho_TB_3, U0, Y_B)
    rho_S_withB = calculate_spin_density_withX(rho_SB, rho_TA_3, U0, Y_A)

    assert np.allclose(abs(np.trace(rho_S_withA @ rho_S_withB)), 1.0, atol=1e-10), \
        f"Fidelity between two states is not 1.0"
    assert np.allclose(rho_S_withA, rho_S_withB, atol=1e-10), \
        f"Spin density matrices are not the same"