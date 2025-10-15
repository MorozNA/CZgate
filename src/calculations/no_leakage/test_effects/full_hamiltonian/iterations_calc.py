import numpy as np
from tqdm import tqdm
from src.y_operator.full_hamiltonian.get_evolution import get_evolution
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.y_operator.params import get_params

# INITIAL PARAMETERS
iterations = 30
n = 20
Q = 0.0
om = 2 * np.pi * 5e6
tau, delta, xi = get_params(om)
t_initial = 0.0
t_final = 2 * tau

# INITIAL STATES
temperature0 = 1e-9
rho_T0 = get_rho_T0(temperature0, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[4, 4] = 1.0
# rho_S0[0, 0] = 1 / 2
# rho_S0[4, 4] = 1 / 2
# rho_S0[0, 4] = 1 / 2
# rho_S0[4, 0] = 1 / 2

rho_full_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))
rho_decomp_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))

rho_ideal = np.copy(rho_S0)

# EVOLUTION OPERATORS
U0 = get_evolution(delta, om, xi, tau, n, Q)

# DATA LISTS
fidelity0 = []
fidelity0_decomp = []


for i in tqdm(range(iterations)):
    rho_full_T0 = exact_evolution(rho_full_T0, U0)
    rho_decomp_T0 = exact_evolution(rho_decomp_T0, U0)

    rho_full_T0_spin = np.einsum('ikjk->ij', rho_full_T0.reshape(9, n**2, 9, n**2))

    rho_decomp_T0_spin = np.einsum('ikjk->ij', rho_decomp_T0.reshape(9, n ** 2, 9, n ** 2))
    rho_decomp_T0_vib = np.einsum('ikil->kl', rho_decomp_T0.reshape(9, n**2, 9, n**2))
    rho_decomp_T0 = np.kron(rho_decomp_T0_spin, rho_decomp_T0_vib)

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    U0_ideal = get_U0_ideal(tau, delta, xi)
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fidelity0.append(generalized_fidelity(rho_full_T0_spin, rho_ideal))
    fidelity0_decomp.append(generalized_fidelity(rho_decomp_T0_spin, rho_ideal))