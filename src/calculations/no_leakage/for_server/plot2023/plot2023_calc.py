import numpy as np
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.algorithm.algorithm_fun import one_iteration, exact_evolution
from src.calculations_no_leakage.for_server.plot2023.plot2023_initial import rho_TA_0, rho_TB_0, rho_S0, Q, n
from tqdm import tqdm

fidelity_iter1 = []

OM1 = 2 * np.pi * 3e6
OM2 = 2 * np.pi * 10e6
om_array = np.linspace(OM1, OM2, 10)

for i_omega in tqdm(range(len(om_array))):
    omega = om_array[i_omega]
    tau = 4.292682 / omega
    t_initial = 0.0
    t_final = 2.0 * tau

    # if i_omega == 0:
    #     print('Tau(0) = ', tau)
    # elif i_omega == len(om_array) - 1:
    #     print('Tau(-1) = ', tau)

    #exact_rho = np.copy(rho_S0)
    # calc_rho = np.copy(rho_S0)

    Y_A = construct_Y_A(t_initial, t_final, omega, Q, n)
    Y_B = construct_Y_B(t_initial, t_final, omega, Q, n)
    U0 = construct_U0(t_final, omega)
    U0_perfect = construct_U0(t_final, omega, False)

    exact_rho = exact_evolution(rho_S0, U0_perfect)
    calc_rho, _, _ = one_iteration(rho_S0, rho_TA_0, rho_TB_0, U0, Y_A, Y_B)

    calc_rho = calc_rho / np.trace(calc_rho)

    fidelity_iter1.append(abs(np.trace(exact_rho @ calc_rho)))

# Save each list/array separately
np.save('om_array_plot2023_40MHz_test.npy', om_array)
np.save('fidelity_iter1_plot2023_40MHz_test.npy', np.array(fidelity_iter1))
