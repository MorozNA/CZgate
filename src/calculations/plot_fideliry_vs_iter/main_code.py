import numpy as np
from src.calculations.algorithm_fun import one_iteration, exact_evolution
from src.calculations.initial_parameters import rho_S0
from src.calculations.initial_parameters import rho_TA_0, rho_TA_1, rho_TA_2, rho_TA_3
from src.calculations.initial_parameters import rho_TB_0, rho_TB_1, rho_TB_2, rho_TB_3
from src.calculations.initial_parameters import U0, U0_perfect, Y_A, Y_B
from tqdm import tqdm

fidelity0 = []
fidelity1 = []
fidelity2 = []
fidelity3 = []
rho_TB_1_list = [np.diag(rho_TB_1)]

rho_S0_0 = np.copy(rho_S0)
rho_S0_1 = np.copy(rho_S0)
rho_S0_2 = np.copy(rho_S0)
rho_S0_3 = np.copy(rho_S0)

iterations = 50

for i in tqdm(range(iterations)):
    exact_rho = exact_evolution(rho_S0, U0_perfect)
    rho_S0 = exact_rho
    calc_rho_0, rho_TA_0, rho_TB_0 = one_iteration(rho_S0_0, rho_TA_0, rho_TB_0, U0, Y_A, Y_B)
    calc_rho_1, rho_TA_1, rho_TB_1 = one_iteration(rho_S0_1, rho_TA_1, rho_TB_1, U0, Y_A, Y_B)
    calc_rho_2, rho_TA_2, rho_TB_2 = one_iteration(rho_S0_2, rho_TA_2, rho_TB_2, U0, Y_A, Y_B)
    calc_rho_3, rho_TA_3, rho_TB_3 = one_iteration(rho_S0_3, rho_TA_3, rho_TB_3, U0, Y_A, Y_B)

    calc_rho_0 = calc_rho_0 / np.trace(calc_rho_0)
    calc_rho_1 = calc_rho_1 / np.trace(calc_rho_1)
    calc_rho_2 = calc_rho_2 / np.trace(calc_rho_2)
    calc_rho_3 = calc_rho_3 / np.trace(calc_rho_3)
    rho_TA_0 = rho_TA_0 / np.trace(rho_TA_0)
    rho_TA_1 = rho_TA_1 / np.trace(rho_TA_1)
    rho_TA_2 = rho_TA_2 / np.trace(rho_TA_2)
    rho_TA_3 = rho_TA_3 / np.trace(rho_TA_3)
    rho_TB_0 = rho_TB_0 / np.trace(rho_TB_0)
    rho_TB_1 = rho_TB_1 / np.trace(rho_TB_1)
    rho_TB_2 = rho_TB_2 / np.trace(rho_TB_2)
    rho_TB_3 = rho_TB_3 / np.trace(rho_TB_3)

    fidelity0.append(abs(np.trace(exact_rho @ calc_rho_0)))
    fidelity1.append(abs(np.trace(exact_rho @ calc_rho_1)))
    fidelity2.append(abs(np.trace(exact_rho @ calc_rho_2)))
    fidelity3.append(abs(np.trace(exact_rho @ calc_rho_3)))

    rho_S0_0 = np.copy(calc_rho_0)
    rho_S0_1 = np.copy(calc_rho_1)
    rho_S0_2 = np.copy(calc_rho_2)
    rho_S0_3 = np.copy(calc_rho_3)

    print('\n')
    print('spin trace: ', abs(np.trace(calc_rho_1)))
    print('vib trace: ', abs(np.trace(rho_TB_1)))

    rho_TB_1_list.append(np.diag(rho_TB_1))

    # if i==75:
    #     rho_iter15_to_save = rho_TB_1
    #     break
    # print('Trace rho_1 = ', np.trace(calc_rho_1))
