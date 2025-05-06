import numpy as np
from tqdm import tqdm
from src.y_operator.params import p0z, HBAR, OM_small, M, lambd_1, lambd_2
from src.calculations.plot_effect.tools_fun import get_rho_T0
from src.y_operator.construct_U0 import construct_U0
# from src.y_operator.tests.bad_u0 import construct_bad_U0
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.calculations.algorithm_fun import one_iteration, exact_evolution

Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
print(HBAR * Q, p0z)
expected_fidelity_flag = True

print('x0: ', np.sqrt(HBAR / M / OM_small))


def get_fidelity_expected(tau_val, q_val):
    x0 = np.sqrt(HBAR / M / OM_small / 2)  # previously 1e-6 was here
    alpha = (HBAR * q_val + p0z) / M * 2 * tau_val / x0
    # alpha = ((q_val * p0z / M) + (HBAR * q_val) ** 2 / 2 / M) * tau_val
    return 1 / 2 * (1 + np.exp(-2 * abs(alpha) ** 2 / 2))


### INITIAL STATES
n = 100
temperature0 = 1e-9
temperature1 = 1e-6
temperature2 = 5e-6
temperature3 = 10e-6
rho_TA_0 = get_rho_T0(temperature0, n)
rho_TA_1 = get_rho_T0(temperature1, n)
rho_TA_2 = get_rho_T0(temperature2, n)
rho_TA_3 = get_rho_T0(temperature3, n)
rho_TB_0 = get_rho_T0(temperature0, n)
rho_TB_1 = get_rho_T0(temperature1, n)
rho_TB_2 = get_rho_T0(temperature2, n)
rho_TB_3 = get_rho_T0(temperature3, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1 / 2
rho_S0[4, 4] = 1 / 2
rho_S0[0, 4] = 1 / 2
rho_S0[4, 0] = 1 / 2

### LISTS TO SAVE DATA
fidelity0 = []
fidelity1 = []
fidelity2 = []
fidelity3 = []
f_expected = []

### ARRAY OF TIMES TAU
tau_array = np.linspace(50, 400, 10) * 1e-9

for tau in tqdm(tau_array):
    omega = 4.292682 / tau

    # print('DELTA: ', 0.377371 * omega)
    # print('addition to DELTA: ', HBAR * (Q ** 2) / 2 / M)
    # print('relation: ', (HBAR * (Q ** 2) / 2 / M) / (0.377371 * omega))

    t_initial = 0.0
    t_final = 2.0 * tau

    Y_A = construct_Y_A(t_initial, t_final, omega, Q, n)
    Y_B = construct_Y_B(t_initial, t_final, omega, Q, n)
    U0 = construct_U0(t_final, omega)

    exact_rho = exact_evolution(rho_S0, U0)
    calc_rho_0, _, _ = one_iteration(rho_S0, rho_TA_0, rho_TB_0, U0, Y_A, Y_B)
    calc_rho_1, _, _ = one_iteration(rho_S0, rho_TA_1, rho_TB_1, U0, Y_A, Y_B)
    calc_rho_2, _, _ = one_iteration(rho_S0, rho_TA_2, rho_TB_2, U0, Y_A, Y_B)
    calc_rho_3, _, _ = one_iteration(rho_S0, rho_TA_3, rho_TB_3, U0, Y_A, Y_B)

    calc_rho_0 = calc_rho_0 / np.trace(calc_rho_0)
    calc_rho_1 = calc_rho_1 / np.trace(calc_rho_1)
    calc_rho_2 = calc_rho_2 / np.trace(calc_rho_2)
    calc_rho_3 = calc_rho_3 / np.trace(calc_rho_3)

    fidelity0.append(abs(np.trace(exact_rho @ calc_rho_0)))
    fidelity1.append(abs(np.trace(exact_rho @ calc_rho_1)))
    fidelity2.append(abs(np.trace(exact_rho @ calc_rho_2)))
    fidelity3.append(abs(np.trace(exact_rho @ calc_rho_3)))
    f_expected.append(get_fidelity_expected(tau, Q))

    # bad_U0 = construct_bad_U0(t_final, omega, Q)
    # expected_fidelity_rho = exact_evolution(rho_S0, bad_U0)
    # f_expected.append(abs(np.trace(exact_rho @ expected_fidelity_rho)))
