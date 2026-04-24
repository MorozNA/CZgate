import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_U0_ideal, exact_evolution, generalized_fidelity
from src.algorithm.algorithm_fun import one_iteration_order2
from src.y_operator.construct_U0 import construct_U0k
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator.config import YOperatorConfig, YOperatorDerived, build_derived
from dataclasses import replace


def calc_optimal_om(cfg: YOperatorConfig, om_left_MHz, om_right_MHz, rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, rho_ideal0, num_of_iter, num_of_omegas, path='data/fidelities/'):
    om_left_hz = om_left_MHz * 1e6
    om_right_hz = om_right_MHz * 1e6

    # DATA LISTS
    om_list = np.linspace(om_left_hz, om_right_hz, num_of_omegas)
    om_list = om_list[::-1]
    tau_list = []
    fidelities = []



    for om in tqdm(om_list):
        # CALCULATION
        cfg_om = replace(cfg, om_hz=om)
        params_om = build_derived(cfg_om)

        # EVOLUTION PROCESS
        U01 = construct_U0k(replace(params_om, xi=0.0), params_om.tau)
        YA1 = construct_Y_A(replace(params_om, xi=0.0), 0.0, params_om.tau)
        YB1 = construct_Y_B(replace(params_om, xi=0.0), 0.0, params_om.tau)

        U02 = construct_U0k(params_om, params_om.tau)
        YA2 = construct_Y_A(params_om, params_om.tau, 2 * params_om.tau)
        YB2 = construct_Y_B(params_om, params_om.tau, 2 * params_om.tau)

        # print('================================')
        # print('F(rho_el_T0,rho_ideal0) = ', generalized_fidelity(rho_el_T0, rho_ideal0))
        rho_elmotA_T, rho_elmotB_T, rho_el_T, rho_T_A, rho_T_B = one_iteration_order2(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U01, YA1, YB1)
        rho_elmotA_T, rho_elmotB_T, rho_el_T, rho_T_A, rho_T_B = one_iteration_order2(rho_elmotA_T, rho_elmotB_T, rho_el_T, rho_T_A, rho_T_B, U02, YA2, YB2)
        rho_el_T /= np.trace(rho_el_T)
        # print('F(rho_el_T0,rho_ideal0) = ', generalized_fidelity(rho_el_T0, rho_ideal0))
        # print('================================')

        # CALCULATING FIDELITY
        U0_ideal = get_U0_ideal(params_om.tau, params_om.delta, params_om.xi)
        rho_ideal = exact_evolution(rho_ideal0, U0_ideal)

        # SAVING INFORMATION
        tau_list.append(params_om.tau)
        fidelities.append(generalized_fidelity(rho_el_T, rho_ideal))


    np.savetxt(path + f'fidelities_{num_of_iter}.txt', fidelities, fmt='%.18f')
    np.savetxt(path + f'omegas_{num_of_iter}.txt', om_list, fmt='%.18f')
    np.savetxt(path + f'taus_{num_of_iter}.txt', tau_list, fmt='%.18f')

    optimal_ind = np.argmax(fidelities)
    optimal_om = 2 * np.pi * om_list[optimal_ind]
    optimal_tau = tau_list[optimal_ind]

    print('___________________________________')
    print('Iteration number: ', num_of_iter)
    print('max fid = ', np.amax(fidelities))
    print('optimal tau = ', optimal_tau)
    print('optimal omega = ', optimal_om)
    print('___________________________________')

    return optimal_om