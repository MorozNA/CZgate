import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.y_operator_deltaR.params import get_params
from src.y_operator_deltaR.full_hamiltonian.get_evolution import get_evolution

# INITIAL PARAMETERS
n = 15
Q = 0.0
# delta_R = 2 * np.pi * 50e6
delta_R = 1e9


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

# DATA LISTS
fidelity0 = []
fidelity1 = []
tau_array = []
om_array = np.linspace(3, 10, 50) * 2 * np.pi * 1e6


# CALCULATIONS
for om in tqdm(om_array):
    tau, delta, xi = get_params(om, delta_R)

    U0 = get_evolution(delta, om, xi, delta_R, tau, n, Q)
    rho_full_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))
    rho_full_T1 = np.kron(rho_S0, np.kron(rho_T1, rho_T1))

    rho_full_T0 = exact_evolution(rho_full_T0, U0).reshape(9, n**2, 9, n**2)
    rho_full_T1 = exact_evolution(rho_full_T1, U0).reshape(9, n**2, 9, n**2)

    rho_full_T0 = np.einsum('ikjk->ij', rho_full_T0)
    rho_full_T1 = np.einsum('ikjk->ij', rho_full_T1)

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    U0_ideal = get_U0_ideal(tau, delta, xi)
    rho_ideal = exact_evolution(rho_S0, U0_ideal)

    tau_array.append(tau)
    fidelity0.append(generalized_fidelity(rho_full_T0, rho_ideal))
    fidelity1.append(generalized_fidelity(rho_full_T1, rho_ideal))
