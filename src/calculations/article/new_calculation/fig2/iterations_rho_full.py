import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.y_operator.full_hamiltonian.get_evolution import get_evolution
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.config import YOperatorConfig, build_derived


# calculation parameters
Q_INT = 1.0
W_INT = 1.0
path = 'data/data_a/'
temperature0 = 1e-6
iterations = 30


# configuration parameters
cfg = YOperatorConfig(
    Q_INT_CONSTANT=Q_INT,
    W_INT_CONSTANT=W_INT,
    om_hz=5e6,
    n=30
)
params = build_derived(cfg)
print(r'$\tau$ = ', params.tau)

# INITIAL STATES
rho_T0 = get_rho_T0(params, temperature0)

rho_S0 = np.zeros((9, 9), dtype=complex)
idx = [0, 1, 3, 4]
rho_S0[np.ix_(idx, idx)] = 1/4 * np.ones((4, 4))

rho_algorithm_T0 = np.copy(rho_S0)
rho_T0_A = get_rho_T0(params, temperature0)
rho_T0_B = get_rho_T0(params, temperature0)

rho_full_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))
rho_decomp_T0 = np.kron(rho_S0, np.kron(rho_T0, rho_T0))

rho_ideal = np.copy(rho_S0)


# EVOLUTION OPERATORS
print(9*params.n*params.n)
U0_full = get_evolution(params)
U0 = construct_U0(params, 2 * params.tau)
U0_ideal = get_U0_ideal(params.tau, params.delta, params.xi)


# DATA LISTS
fidelity_full = []
fidelity_full_decomp = []
purity = []


for i in tqdm(range(iterations)):
    rho_full_T0 = exact_evolution(rho_full_T0, U0_full)
    rho_decomp_T0 = exact_evolution(rho_decomp_T0, U0_full)

    rho_full_T0_spin = np.einsum('ikjk->ij', rho_full_T0.reshape(9, params.n**2, 9, params.n**2))

    rho_decomp_T0_spin = np.einsum('ikjk->ij', rho_decomp_T0.reshape(9, params.n ** 2, 9, params.n ** 2))
    rho_decomp_T0_vib = np.einsum('ikil->kl', rho_decomp_T0.reshape(9, params.n**2, 9, params.n**2))
    rho_decomp_T0 = np.kron(rho_decomp_T0_spin, rho_decomp_T0_vib)

    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fidelity_full.append(generalized_fidelity(rho_full_T0_spin, rho_ideal))
    fidelity_full_decomp.append(generalized_fidelity(rho_decomp_T0_spin, rho_ideal))

np.savetxt(path + 'fidelity_full.txt', fidelity_full, fmt='%.18f')
np.savetxt(path + 'fidelity_full_decomp.txt', fidelity_full_decomp, fmt='%.18f')