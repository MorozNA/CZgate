import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.algorithm.other_tools import construct_U0_for_trotter
from src.algorithm.algorithm_fun_other import one_iteration
from src.y_operator_deltaR.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator_deltaR.params import lambd_1, lambd_2
from src.y_operator_deltaR.params import get_params


temp_name = 1
path = 'data/data_c/'


# INITIAL PARAMETERS
temperature0 = 1e-6
Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
# Q = 0.0
delta_R = 2 * np.pi * 50e6

n = 30
iterations = 30

om = 2 * np.pi * 5e6
tau, delta, xi = get_params(om, delta_R)
print(r'$\tau$ = ', tau)

# INITIAL STATES
rho_T0 = get_rho_T0(temperature0, n)
# rho_T0 = np.zeros((n, n), dtype=complex)
# rho_T0[0, 0] = 1.0

rho_S0 = np.zeros((9, 9), dtype=complex)
idx = [0, 1, 3, 4]
rho_S0[np.ix_(idx, idx)] = 1/4 * np.ones((4, 4))

rho_elmotA_T0 = np.kron(rho_S0, rho_T0)
rho_elmotB_T0 = np.kron(rho_S0, rho_T0)
rho_el_T0 = np.copy(rho_S0)
rho_T0_A = np.copy(rho_T0)
rho_T0_B = np.copy(rho_T0)

rho_ideal = np.copy(rho_S0)


# EVOLUTION OPERATORS
U01 = construct_U0_for_trotter(tau, om, tau, delta, 0.0, delta_R)
YA1 = construct_Y_A(0.0, tau, om, tau, delta, 0.0, delta_R, Q, n)
YB1 = construct_Y_B(0.0, tau, om, tau, delta, 0.0, delta_R, Q, n)

U02 = construct_U0_for_trotter(tau, om, tau, delta, xi, delta_R)
YA2 = construct_Y_A(tau, 2 * tau, om, tau, delta, xi, delta_R, Q, n)
YB2 = construct_Y_B(tau, 2 * tau, om, tau, delta, xi, delta_R, Q, n)


# DATA LISTS
fidelity_hybrid = []


for i in tqdm(range(iterations)):
    _, _, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U01, YA1, YB1)
    rho_elmotA_T0 = np.kron(rho_el_T0, rho_T0_A)
    rho_elmotB_T0 = np.kron(rho_el_T0, rho_T0_B)
    _, _, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U02, YA2, YB2)
    rho_elmotA_T0 = np.kron(rho_el_T0, rho_T0_A)
    rho_elmotB_T0 = np.kron(rho_el_T0, rho_T0_B)

    rho_elmotA_T0 /= np.trace(rho_elmotA_T0)
    rho_elmotB_T0 /= np.trace(rho_elmotB_T0)
    rho_el_T0 /= np.trace(rho_el_T0)
    rho_T0_A /= np.trace(rho_T0_A)
    rho_T0_B /= np.trace(rho_T0_B)


    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    U0_ideal = get_U0_ideal(tau, delta, xi)
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fidelity_hybrid.append(generalized_fidelity(rho_el_T0, rho_ideal))

np.savetxt(path + f'fidelity_hybrid_pulse_{temp_name}.txt', fidelity_hybrid, fmt='%.18f')