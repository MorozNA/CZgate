import numpy as np
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.calculations.algorithm_fun import one_iteration, exact_evolution
from src.calculations.for_server.distribution_change.temp1_distr_init import rho_TA_0, rho_TB_0, rho_S0, Q, n
from tqdm import tqdm

rho_TA_list = [np.diag(rho_TA_0)]
rho_TB_list = [np.diag(rho_TB_0)]

iterations = 200
tau = 800e-9

omega = 4.292682 / tau
t_initial = 0.0
t_final = 2.0 * tau

exact_rho = np.copy(rho_S0)
calc_rho = np.copy(rho_S0)
rho_TA = np.copy(rho_TA_0)
rho_TB = np.copy(rho_TB_0)

Y_A = construct_Y_A(t_initial, t_final, omega, Q, n)
Y_B = construct_Y_B(t_initial, t_final, omega, Q, n)
U0 = construct_U0(t_final, omega)
U0_perfect = construct_U0(t_final, omega, False)

for i in tqdm(range(iterations)):
    exact_rho = exact_evolution(exact_rho, U0_perfect)
    calc_rho, rho_TA, rho_TB = one_iteration(calc_rho, rho_TA, rho_TB, U0, Y_A, Y_B)

    calc_rho = calc_rho / np.trace(calc_rho)
    rho_TA = rho_TA / np.trace(rho_TA)
    rho_TB = rho_TB / np.trace(rho_TB)

    rho_TA_list.append(np.diag(rho_TA))
    rho_TB_list.append(np.diag(rho_TB))

# Save each list/array separately
np.save('plotting_codes/temp1_rho_TA_list.npy', np.array(rho_TA_list))
np.save('plotting_codes/temp1_rho_TB_list.npy', np.array(rho_TB_list))
