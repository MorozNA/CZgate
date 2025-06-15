import numpy as np
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.calculations.algorithm_fun import one_iteration, exact_evolution
from src.calculations.for_server.temp0.temp0_initial import rho_TA_0, rho_TB_0, rho_S0, Q, n
from tqdm import tqdm

fidelity_iter1 = []
fidelity_iter5 = []
fidelity_iter10 = []
fidelity_iter15 = []
fidelity_iter20 = []
fidelity_iter25 = []
rho_TA_tau400_list = [np.diag(rho_TA_0)]  # chosen to have a close OM with 2022 publication
rho_TB_tau400_list = [np.diag(rho_TB_0)]

iterations = 20
tau_array = np.linspace(200, 1400, 300) * 1e-9

for i_tau in tqdm(range(len(tau_array))):
    tau = tau_array[i_tau]
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

    for i in range(iterations):

        exact_rho = exact_evolution(exact_rho, U0_perfect)
        calc_rho, rho_TA, rho_TB = one_iteration(calc_rho, rho_TA, rho_TB, U0, Y_A, Y_B)

        calc_rho = calc_rho / np.trace(calc_rho)
        rho_TA = rho_TA / np.trace(rho_TA)
        rho_TB = rho_TB / np.trace(rho_TB)

        if i == 0:
            fidelity_iter1.append(abs(np.trace(exact_rho @ calc_rho)))
        elif i == 4:
            fidelity_iter5.append(abs(np.trace(exact_rho @ calc_rho)))
        elif i == 9:
            fidelity_iter10.append(abs(np.trace(exact_rho @ calc_rho)))
        elif i == 14:
            fidelity_iter15.append(abs(np.trace(exact_rho @ calc_rho)))
        elif i == 19:
            fidelity_iter20.append(abs(np.trace(exact_rho @ calc_rho)))
        elif i == 24:
            fidelity_iter25.append(abs(np.trace(exact_rho @ calc_rho)))
        if i_tau == (len(tau_array) // 2):
            rho_TA_tau400_list.append(np.diag(rho_TA))
            rho_TB_tau400_list.append(np.diag(rho_TB))

# Save each list/array separately
np.save('temp0_tau.npy', tau_array)
np.save('temp0_fidelity_iter1.npy', np.array(fidelity_iter1))
np.save('temp0_fidelity_iter5.npy', np.array(fidelity_iter5))
np.save('temp0_fidelity_iter10.npy', np.array(fidelity_iter10))
np.save('temp0_fidelity_iter15.npy', np.array(fidelity_iter15))
np.save('temp0_fidelity_iter20.npy', np.array(fidelity_iter20))
np.save('temp0_fidelity_iter25.npy', np.array(fidelity_iter25))
np.save('temp0_rho_TA_tau400_list.npy', np.array(rho_TA_tau400_list))
np.save('temp0_rho_TB_tau400_list.npy', np.array(rho_TB_tau400_list))