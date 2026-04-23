import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.algorithm.algorithm_fun_other import one_iteration
from src.algorithm.other_tools import construct_U0_for_trotter
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator.params import lambd_1, lambd_2
from src.y_operator.calc_params import calc_params


path = 'data/data_a/opt_omega_30/'


# INITIAL PARAMETERS
temperature = 5e-6
# om = 2 * np.pi * 5e6
# om = 2 * np.pi * 5.74457429048414 * 1e6  # 5 muK, n = 1
om = 2 * np.pi * 6.6701168614357265 * 1e6  # 5 muK, n = 30

n = 200
iterations = 50

Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
# Q = 0.0
delta_R = 2 * np.pi * 50e6

tau, delta, xi = calc_params(om, delta_R)
t_initial = 0.0
t_final = 2 * tau

print('Om = ', om / 2 / np.pi / 1e6, 'MHz')
print('Tau = ', tau / 1e-9, ' ns')


# INITIAL STATES
rho_T0 = get_rho_T0(temperature, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
idx = [0, 1, 3, 4]
rho_S0[np.ix_(idx, idx)] = 1/len(idx) * np.ones((len(idx), len(idx)))

rho_elmotA_T0 = np.kron(rho_S0, rho_T0)
rho_elmotB_T0 = np.kron(rho_S0, rho_T0)
rho_el_T0 = np.copy(rho_S0)
rho_T0_A = get_rho_T0(temperature, n)
rho_T0_B = get_rho_T0(temperature, n)

rho_ideal = np.copy(rho_S0)


# EVOLUTION OPERATORS
print("=================================================================")
U0_ideal = get_U0_ideal(tau, delta, xi)
U01 = construct_U0_for_trotter(tau, om, tau, delta, 0.0, delta_R)
YA1 = construct_Y_A(0.0, tau, om, tau, delta, 0.0, Q, n, delta_R)
YB1 = construct_Y_B(0.0, tau, om, tau, delta, 0.0, Q, n, delta_R)

U02 = construct_U0_for_trotter(tau, om, tau, delta, xi, delta_R)
YA2 = construct_Y_A(tau, 2 * tau, om, tau, delta, xi, Q, n, delta_R)
YB2 = construct_Y_B(tau, 2 * tau, om, tau, delta, xi, Q, n, delta_R)
print("=================================================================")


# DATA LISTS
fidelities = []


for i in tqdm(range(iterations)):
    rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U01, YA1, YB1)
    rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U02, YA2, YB2)
    rho_elmotA_T0 /= np.trace(rho_elmotA_T0)
    rho_elmotB_T0 /= np.trace(rho_elmotB_T0)
    rho_el_T0 /= np.trace(rho_el_T0)
    rho_T0_A /= np.trace(rho_T0_A)
    rho_T0_B /= np.trace(rho_T0_B)

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    U0_ideal = get_U0_ideal(tau, delta, xi)
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fidelities.append(generalized_fidelity(rho_el_T0, rho_ideal))



np.savetxt(path + f'fidelity_{int(temperature / 1e-6)}.txt', fidelities, fmt='%.18f')