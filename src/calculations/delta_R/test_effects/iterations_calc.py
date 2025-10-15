import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.algorithm.algorithm_fun import one_iteration
from src.y_operator_deltaR.full_hamiltonian.get_evolution import get_evolution
from src.y_operator_deltaR.construct_U0 import construct_U0
from src.y_operator_deltaR.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator_deltaR.params import lambd_1, lambd_2
from src.y_operator_deltaR.params import get_params


path = 'data/data_nothing/temp1/'


# INITIAL PARAMETERS
temperature0 = 1e-6
Q = 0.0
# Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
delta_R = 2 * np.pi * 50e6

# n = 30
n = 20
iterations = 30

om = 2 * np.pi * 5e6
tau, delta, xi = get_params(om, delta_R)
t_initial = 0.0
t_final = 2 * tau
print(tau)

# INITIAL STATES
rho_T0 = get_rho_T0(temperature0, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[4, 4] = 1.0

rho_algorithm_T0 = np.copy(rho_S0)
rho_T0_A = get_rho_T0(temperature0, n)
rho_T0_B = get_rho_T0(temperature0, n)

rho_full_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))
rho_decomp_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))

rho_ideal = np.copy(rho_S0)


# EVOLUTION OPERATORS
U0_full = get_evolution(delta, om, xi, delta_R, tau, n, Q)
U0 = construct_U0(t_final, om, tau, delta, xi, delta_R)
YA = construct_Y_A(t_initial, t_final, om, tau, delta, xi, delta_R, Q, n)
YB = construct_Y_B(t_initial, t_final, om, tau, delta, xi, delta_R, Q, n)


# DATA LISTS
fidelity_alg = []
fidelity_full = []
fidelity_full_decomp = []
purity = []


for i in tqdm(range(iterations)):
    rho_algorithm_T0, rho_T0_A, rho_T0_B = one_iteration(rho_algorithm_T0, rho_T0_A, rho_T0_B, U0, YA, YB)
    rho_full_T0 = exact_evolution(rho_full_T0, U0_full)
    rho_decomp_T0 = exact_evolution(rho_decomp_T0, U0_full)

    rho_algorithm_T0 = rho_algorithm_T0 / np.trace(rho_algorithm_T0)
    rho_T0_A = rho_T0_A / np.trace(rho_T0_A)
    rho_T0_B = rho_T0_B / np.trace(rho_T0_B)


    rho_full_T0_spin = np.einsum('ikjk->ij', rho_full_T0.reshape(9, n**2, 9, n**2))

    rho_decomp_T0_spin = np.einsum('ikjk->ij', rho_decomp_T0.reshape(9, n ** 2, 9, n ** 2))
    rho_decomp_T0_vib = np.einsum('ikil->kl', rho_decomp_T0.reshape(9, n**2, 9, n**2))
    rho_decomp_T0 = np.kron(rho_decomp_T0_spin, rho_decomp_T0_vib)

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    U0_ideal = get_U0_ideal(tau, delta, xi)
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fidelity_alg.append(generalized_fidelity(rho_algorithm_T0, rho_ideal))
    fidelity_full.append(generalized_fidelity(rho_full_T0_spin, rho_ideal))
    fidelity_full_decomp.append(generalized_fidelity(rho_decomp_T0_spin, rho_ideal))
    purity.append(abs(np.trace(rho_full_T0_spin @ rho_full_T0_spin)))

np.savetxt(path + 'fidelity_alg.txt', fidelity_alg, fmt='%.18f')
np.savetxt(path + 'fidelity_full.txt', fidelity_full, fmt='%.18f')
np.savetxt(path + 'fidelity_full_decomp.txt', fidelity_full_decomp, fmt='%.18f')
np.savetxt(path + 'purity.txt', purity, fmt='%.18f')