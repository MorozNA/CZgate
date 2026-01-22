import numpy as np
from tqdm import tqdm
from src.y_operator_deltaR.full_hamiltonian.get_evolution import get_evolution
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.algorithm.algorithm_fun import one_iteration
from src.y_operator_deltaR.params import get_params

# INITIAL PARAMETERS
iterations = 30
n = 20
Q = 0.0
# delta_R = 2 * np.pi * 50e6
delta_R = 2 * np.pi * 100e6
om = 2 * np.pi * 5e6
tau, delta, xi = get_params(om, delta_R)
t_initial = 0.0
t_final = 2 * tau

# INITIAL STATES
temperature0 = 1e-9
temperature1 = 1e-6
rho_T0 = get_rho_T0(temperature0, n)
rho_T1 = get_rho_T0(temperature1, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1 / 2
rho_S0[4, 4] = 1 / 2
rho_S0[0, 4] = 1 / 2
rho_S0[4, 0] = 1 / 2

rho_full_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))
rho_full_T1 = np.kron(rho_S0, np.kron(rho_T1, rho_T1))

rho_ideal = np.copy(rho_S0)

# EVOLUTION OPERATORS
U0 = get_evolution(delta, om, xi, delta_R, tau, n, Q)

# DATA LISTS
fidelity0 = []
fidelity1 = []

for i in tqdm(range(iterations)):
    rho_full_T0 = exact_evolution(rho_full_T0, U0)
    rho_full_T1 = exact_evolution(rho_full_T1, U0)

    rho_full_T0_spin = np.einsum('ikjk->ij', rho_full_T0.reshape(9, n**2, 9, n**2))
    rho_full_T1_spin = np.einsum('ikjk->ij', rho_full_T1.reshape(9, n**2, 9, n**2))

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    U0_ideal = get_U0_ideal(tau, delta, xi)
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fidelity0.append(generalized_fidelity(rho_full_T0_spin, rho_ideal))
    fidelity1.append(generalized_fidelity(rho_full_T1_spin, rho_ideal))