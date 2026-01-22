import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.y_operator_deltaR.construct_U0 import construct_U0
from src.y_operator_deltaR.construct_Y import construct_Y_A, construct_Y_B
from src.algorithm.algorithm_fun import one_iteration
from src.y_operator_deltaR.params import get_params


# INITIAL PARAMETERS
n = 200
Q = 0.0
delta_R = 2 * np.pi * 50e6
# delta_R = 1e9


# INITIAL STATES
temperature0 = 1e-9
temperature1 = 1e-6
temperature5 = 5e-6
rho_T0 = get_rho_T0(temperature0, n)
rho_T1 = get_rho_T0(temperature1, n)
rho_T5 = get_rho_T0(temperature5, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1 / 2
rho_S0[4, 4] = 1 / 2
rho_S0[0, 4] = 1 / 2
rho_S0[4, 0] = 1 / 2

# DATA LISTS
fidelity0 = []
fidelity1 = []
fidelity5 = []
om_array = np.linspace(3, 10, 50) * 2 * np.pi * 1e6
tau_array = []


# CALCULATIONS
for om in tqdm(om_array):
    tau, delta, xi = get_params(om, delta_R)
    t_initial = 0.0
    t_final = 2 * tau

    # ALGORITHM
    U0 = construct_U0(t_final, om, tau, delta, xi, delta_R)
    print('\n')
    print('1')
    YA = construct_Y_A(t_initial, t_final, om, tau, delta, xi, delta_R, Q, n)
    YB = construct_Y_B(t_initial, t_final, om, tau, delta, xi, delta_R, Q, n)
    print('2', '\n')

    rho_algorithm_T0, _, _ = one_iteration(np.copy(rho_S0), np.copy(rho_T0), np.copy(rho_T0), U0, YA, YB)
    rho_algorithm_T1, _, _ = one_iteration(np.copy(rho_S0), np.copy(rho_T1), np.copy(rho_T1), U0, YA, YB)
    rho_algorithm_T5, _, _ = one_iteration(np.copy(rho_S0), np.copy(rho_T5), np.copy(rho_T5), U0, YA, YB)
    print('3', '\n')

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    U0_ideal = get_U0_ideal(tau, delta, xi)
    rho_ideal = exact_evolution(rho_S0, U0_ideal)
    print('4', '\n')

    # Save info
    tau_array.append(tau)
    fidelity0.append(generalized_fidelity(rho_algorithm_T0, rho_ideal))
    fidelity1.append(generalized_fidelity(rho_algorithm_T1, rho_ideal))
    fidelity5.append(generalized_fidelity(rho_algorithm_T5, rho_ideal))