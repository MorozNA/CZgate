import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.y_operator_deltaR.full_hamiltonian.get_evolution import get_evolution
from src.y_operator_deltaR.construct_U0 import construct_U0
from src.y_operator_deltaR.params import lambd_1, lambd_2
from src.y_operator_deltaR.params import get_params


path = 'data/data_b/'


# INITIAL PARAMETERS
temperature0 = 1e-6
Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
# Q = 0.0
delta_R = 2 * np.pi * 50e6

n = 30
iterations = 30

om = 2 * np.pi * 5e6
tau, delta, xi = get_params(om, delta_R)
t_initial = 0.0
t_final = 2 * tau
print(r'$\tau$ = ', tau)

# INITIAL STATES
rho_T0 = get_rho_T0(temperature0, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
idx = [0, 1, 3, 4]
rho_S0[np.ix_(idx, idx)] = 1/4 * np.ones((4, 4))

rho_algorithm_T0 = np.copy(rho_S0)
rho_T0_A = get_rho_T0(temperature0, n)
rho_T0_B = get_rho_T0(temperature0, n)

rho_full_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))
rho_decomp_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))

rho_ideal = np.copy(rho_S0)


# EVOLUTION OPERATORS
print(9*n*n)
U0_full = get_evolution(delta, om, xi, delta_R, tau, n, Q)
U0 = construct_U0(t_final, om, tau, delta, xi, delta_R)


# DATA LISTS
fidelity_full = []
fidelity_full_decomp = []
purity = []


for i in tqdm(range(iterations)):
    rho_full_T0 = exact_evolution(rho_full_T0, U0_full)
    rho_decomp_T0 = exact_evolution(rho_decomp_T0, U0_full)

    rho_full_T0_spin = np.einsum('ikjk->ij', rho_full_T0.reshape(9, n**2, 9, n**2))

    rho_decomp_T0_spin = np.einsum('ikjk->ij', rho_decomp_T0.reshape(9, n ** 2, 9, n ** 2))
    rho_decomp_T0_vib = np.einsum('ikil->kl', rho_decomp_T0.reshape(9, n**2, 9, n**2))
    rho_decomp_T0 = np.kron(rho_decomp_T0_spin, rho_decomp_T0_vib)

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    U0_ideal = get_U0_ideal(tau, delta, xi)
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fidelity_full.append(generalized_fidelity(rho_full_T0_spin, rho_ideal))
    fidelity_full_decomp.append(generalized_fidelity(rho_decomp_T0_spin, rho_ideal))

np.savetxt(path + 'fidelity_full.txt', fidelity_full, fmt='%.18f')
np.savetxt(path + 'fidelity_full_decomp.txt', fidelity_full_decomp, fmt='%.18f')