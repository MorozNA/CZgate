import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_U0_ideal, exact_evolution, generalized_fidelity
from src.algorithm.algorithm_fun_other import one_iteration
from src.algorithm.other_tools import construct_U0_for_trotter
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator.params import get_params


def calc_optimal_om(om_left_MHz, om_right_MHz, Q, rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, rho_ideal0, num_of_iter, num_of_omegas, delta_R=None, path='data/fidelities/'):

    n = len(rho_T0_A)
    om_left = 2 * np.pi * om_left_MHz * 1e6
    om_right = 2 * np.pi * om_right_MHz * 1e6


    # DATA LISTS
    om_list = np.linspace(om_left, om_right, num_of_omegas)
    om_list = om_list[::-1]
    tau_list = []
    fidelities = []



    for om in tqdm(om_list):
        # CALCULATION
        tau, delta, xi = get_params(om, delta_R)

        # EVOLUTION PROCESS
        U01 = construct_U0_for_trotter(tau, om, tau, delta, 0.0, delta_R)
        YA1 = construct_Y_A(0.0, tau, om, tau, delta, 0.0, Q, n, delta_R)
        YB1 = construct_Y_B(0.0, tau, om, tau, delta, 0.0, Q, n, delta_R)

        U02 = construct_U0_for_trotter(tau, om, tau, delta, xi, delta_R)
        YA2 = construct_Y_A(tau, 2 * tau, om, tau, delta, xi, Q, n, delta_R)
        YB2 = construct_Y_B(tau, 2 * tau, om, tau, delta, xi, Q, n, delta_R)

        # print('================================')
        # print('F(rho_el_T0,rho_ideal0) = ', generalized_fidelity(rho_el_T0, rho_ideal0))
        rho_elmotA_T, rho_elmotB_T, rho_el_T, rho_T_A, rho_T_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U01, YA1, YB1)
        rho_elmotA_T, rho_elmotB_T, rho_el_T, rho_T_A, rho_T_B = one_iteration(rho_elmotA_T, rho_elmotB_T, rho_el_T, rho_T_A, rho_T_B, U02, YA2, YB2)
        rho_el_T /= np.trace(rho_el_T)
        # print('F(rho_el_T0,rho_ideal0) = ', generalized_fidelity(rho_el_T0, rho_ideal0))
        # print('================================')

        # CALCULATING FIDELITY
        U0_ideal = get_U0_ideal(tau, delta, xi)
        rho_ideal = exact_evolution(rho_ideal0, U0_ideal)

        # SAVING INFORMATION
        tau_list.append(tau)
        fidelities.append(generalized_fidelity(rho_el_T, rho_ideal))


    np.savetxt(path + f'fidelities_{num_of_iter}.txt', fidelities, fmt='%.18f')
    np.savetxt(path + f'omegas_{num_of_iter}.txt', om_list, fmt='%.18f')
    np.savetxt(path + f'taus_{num_of_iter}.txt', tau_list, fmt='%.18f')

    optimal_ind = np.argmax(fidelities)
    optimal_om = om_list[optimal_ind]
    optimal_tau = tau_list[optimal_ind]

    print('___________________________________')
    print('Iteration number: ', num_of_iter)
    print('max fid = ', np.amax(fidelities))
    print('optimal tau = ', optimal_tau)
    print('optimal omega = ', optimal_om)
    print('___________________________________')

    return optimal_om