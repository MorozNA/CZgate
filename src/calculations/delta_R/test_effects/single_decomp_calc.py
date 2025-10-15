import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.algorithm.algorithm_fun import one_iteration
from src.y_operator_deltaR.construct_U0 import construct_U0
from src.y_operator_deltaR.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator_deltaR.params import lambd_1, lambd_2
from src.y_operator_deltaR.params import get_params


path = 'data/data_QWL/temp1/n_30/'


# INITIAL PARAMETERS
temperature0 = 1e-6
# Q = 0.0
Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
delta_R = 2 * np.pi * 50e6

n = 20
iterations = 30

om = 2 * np.pi * 5e6
tau, delta, xi = get_params(om, delta_R)
t_initial = 0.0
t_final = 2 * tau

# INITIAL STATES
rho_T0 = get_rho_T0(temperature0, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[4, 4] = 1.0

rho_T0_A = get_rho_T0(temperature0, n)
rho_T0_B = get_rho_T0(temperature0, n)



# EVOLUTION OPERATORS
U0 = construct_U0(t_final, om, tau, delta, xi, delta_R)
YA = construct_Y_A(t_initial, t_final, om, tau, delta, xi, delta_R, Q, n)
YB = construct_Y_B(t_initial, t_final, om, tau, delta, xi, delta_R, Q, n)

rho_ideal = np.copy(rho_S0)
U0_ideal = get_U0_ideal(tau, delta, xi)


# DATA LISTS
fidelity_single_decomp = []


for i in tqdm(range(iterations)):
    print(i, '\n')
    rho_algorithm_T0 = np.copy(rho_S0)

    m_pow = i + 1
    rho_algorithm, _, _ = one_iteration(rho_algorithm_T0, rho_T0_A, rho_T0_B, np.linalg.matrix_power(U0, m_pow), np.linalg.matrix_power(YA, m_pow), np.linalg.matrix_power(YB, m_pow))

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fid = generalized_fidelity(rho_algorithm, rho_ideal)
    fidelity_single_decomp.append(fid)
    print(fid)

np.savetxt(path + 'fidelity_single_decomp.txt', fidelity_single_decomp, fmt='%.18f')