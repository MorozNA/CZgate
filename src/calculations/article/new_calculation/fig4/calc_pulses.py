import os
import numpy as np
from src.algorithm.other_tools import get_U0_ideal, exact_evolution, get_rho_T0, construct_U0_for_trotter
from calc_optimal_func import calc_optimal_om
from src.algorithm.algorithm_fun_other import one_iteration
from src.y_operator_deltaR.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator_deltaR.params import lambd_1, lambd_2, get_params


T_muK = 5
n = 200
om_left_MHz = 2.4
om_right_MHz = 15
num_of_iter = 25
path = f'data/fidelities/T{T_muK}/n{n}/om_{om_left_MHz}_{om_right_MHz}/'
os.makedirs(path, exist_ok=True)

num_omegas = 600


temperature = T_muK * 1e-6
Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
delta_R = 2 * np.pi * 50e6

rho_S0 = np.zeros((9, 9), dtype=complex)
idx = [0, 1, 3, 4]
rho_S0[np.ix_(idx, idx)] = 1/4 * np.ones((4, 4))


if T_muK==0:
    rho_T0 = np.zeros(n, dtype=complex)
    rho_T0[0, 0] = 1.0
else:
    rho_T0 = get_rho_T0(temperature, n)


# om0 = 2 * np.pi * 2.5e6
rho_elmotA_T0 = np.kron(rho_S0, rho_T0)  # var
rho_elmotB_T0 = np.kron(rho_S0, rho_T0)  # var
rho_el_T0 = np.copy(rho_S0)  # var
rho_T0_A = np.copy(rho_T0)  # var
rho_T0_B = np.copy(rho_T0)  # var
rho_ideal0 = np.copy(rho_S0)  # var


if num_of_iter > 1:
    opt_ind = np.argmax(np.loadtxt(path + f'fidelities_{1}.txt'))
    om0 = np.loadtxt(path + f'omegas_{1}.txt')[opt_ind]
    print('om0 = ', om0 / (2 * np.pi * 1e6))

    tau, delta, xi = get_params(om0, delta_R)
    U0_ideal = get_U0_ideal(tau, delta, xi)
    U01 = construct_U0_for_trotter(tau, om0, tau, delta, 0.0, delta_R)
    YA1 = construct_Y_A(0.0, tau, om0, tau, delta, 0.0, delta_R, Q, n)
    YB1 = construct_Y_B(0.0, tau, om0, tau, delta, 0.0, delta_R, Q, n)
    U02 = construct_U0_for_trotter(tau, om0, tau, delta, xi, delta_R)
    YA2 = construct_Y_A(tau, 2 * tau, om0, tau, delta, xi, delta_R, Q, n)
    YB2 = construct_Y_B(tau, 2 * tau, om0, tau, delta, xi, delta_R, Q, n)
    for i in range(num_of_iter-1):
        print('iter = ', i + 1, '\n')
        rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U01, YA1, YB1)
        rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U02, YA2, YB2)
        rho_elmotA_T0 /= np.trace(rho_elmotA_T0)
        rho_elmotB_T0 /= np.trace(rho_elmotB_T0)
        rho_el_T0 /= np.trace(rho_el_T0)
        rho_T0_A /= np.trace(rho_T0_A)
        rho_T0_B /= np.trace(rho_T0_B)
        rho_ideal0 = exact_evolution(rho_ideal0, U0_ideal)


optimal_om = calc_optimal_om(om_left_MHz, om_right_MHz, Q, delta_R, rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, rho_ideal0, num_of_iter, num_omegas, path)
print(optimal_om / 2 / np.pi / 1e6)