import os
import numpy as np
from src.algorithm.other_tools import get_U0_ideal, exact_evolution, get_rho_T0
from calc_optimal_func import calc_optimal_om
from src.algorithm.algorithm_fun import one_iteration_order2
from src.y_operator.construct_U0 import construct_U0k
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator.config import YOperatorConfig, build_derived
from dataclasses import replace


# calculation parameters
T_muK = 0.5
temperature0 = T_muK * 1e-6
n = 50
om_left_MHz = 2.4
om_right_MHz = 15
num_of_iter = 50
path = f'data/fidelities/T{T_muK}/n{n}/om_{om_left_MHz}_{om_right_MHz}/'
os.makedirs(path, exist_ok=True)
num_omegas = 600


# configuration parameters
cfg_iter = YOperatorConfig(
    delta_rydberg_hz=50e6,
    n=n
)
params_iter = build_derived(cfg_iter)

rho_S0 = np.zeros((9, 9), dtype=complex)
idx = [0, 1, 3, 4]
rho_S0[np.ix_(idx, idx)] = 1/4 * np.ones((4, 4))


if T_muK==0:
    rho_T0 = np.zeros(n, dtype=complex)
    rho_T0[0, 0] = 1.0
else:
    rho_T0 = get_rho_T0(params_iter, temperature0)


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

    cfg_iter = replace(cfg_iter, om_hz=float(om0 / (2 * np.pi)))
    params_iter = build_derived(cfg_iter)
    U0_ideal = get_U0_ideal(params_iter.tau, params_iter.delta, params_iter.xi)
    U01 = construct_U0k(replace(params_iter, xi=0.0), params_iter.tau)
    YA1 = construct_Y_A(replace(params_iter, xi=0.0), 0.0, params_iter.tau)
    YB1 = construct_Y_B(replace(params_iter, xi=0.0), 0.0, params_iter.tau)
    U02 = construct_U0k(params_iter, params_iter.tau)
    YA2 = construct_Y_A(params_iter, params_iter.tau, 2 * params_iter.tau)
    YB2 = construct_Y_B(params_iter, params_iter.tau, 2 * params_iter.tau)
    for i in range(num_of_iter-1):
        print('iter = ', i + 1, '\n')
        rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration_order2(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U01, YA1, YB1)
        rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration_order2(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U02, YA2, YB2)
        rho_elmotA_T0 /= np.trace(rho_elmotA_T0)
        rho_elmotB_T0 /= np.trace(rho_elmotB_T0)
        rho_el_T0 /= np.trace(rho_el_T0)
        rho_T0_A /= np.trace(rho_T0_A)
        rho_T0_B /= np.trace(rho_T0_B)
        rho_ideal0 = exact_evolution(rho_ideal0, U0_ideal)


optimal_om = calc_optimal_om(cfg_iter, om_left_MHz, om_right_MHz, rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, rho_ideal0, num_of_iter, num_omegas, path)
print(optimal_om / 2 / np.pi / 1e6)