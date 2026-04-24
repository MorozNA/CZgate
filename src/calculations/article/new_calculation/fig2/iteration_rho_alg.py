import numpy as np
from tqdm import tqdm
from src.algorithm.other_tools import get_rho_T0, get_U0_ideal, exact_evolution, generalized_fidelity
from src.algorithm.algorithm_fun import one_iteration_order2
from src.y_operator.construct_U0 import construct_U0k
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator.config import YOperatorConfig, build_derived
from dataclasses import replace


# calculation parameters
temperature0 = 1e-6
iterations = 30
temp_name = 1
path = 'data/data_a/'


# configuration parameters
cfg = YOperatorConfig(
    DELTA_a_hz=-0.7e6,
    DELTA_b_hz=-2.5e6,
    DELTA_r_hz=-2.5e6,

    W_INT_CONSTANT=1.0,
    w01=10e-6,
    w02=10e-6,

    Q_INT_CONSTANT=1.0,
    lambd_1=795e-9,
    lambd_2=480e-9,
    OM_small_hz=7.158e3,

    om_hz=5e6,
    delta_rydberg_hz=50e6,
    n=30
)
params = build_derived(cfg)


# INITIAL STATES
rho_S0 = np.zeros((9, 9), dtype=complex)
idx = [0, 1, 3, 4]
rho_S0[np.ix_(idx, idx)] = 1/4 * np.ones((4, 4))

rho_T0 = get_rho_T0(params, temperature0)

rho_elmotA_T0 = np.kron(rho_S0, rho_T0)
rho_elmotB_T0 = np.kron(rho_S0, rho_T0)
rho_el_T0 = np.copy(rho_S0)
rho_T0_A = np.copy(rho_T0)
rho_T0_B = np.copy(rho_T0)

rho_ideal = np.copy(rho_S0)


# EVOLUTION OPERATORS
print("=================================================================")
# TODO: rewrite U0_ideal function
U0_ideal = get_U0_ideal(params.tau, params.delta, params.xi)

U01 = construct_U0k(replace(params, xi=0.0), params.tau)
YA1 = construct_Y_A(replace(params, xi=0.0), 0.0, params.tau)
YB1 = construct_Y_B(replace(params, xi=0.0), 0.0, params.tau)

U02 = construct_U0k(params, params.tau)
YA2 = construct_Y_A(params, params.tau, 2 * params.tau)
YB2 = construct_Y_B(params, params.tau, 2 * params.tau)
print("=================================================================")


# DATA LISTS
fidelity_alg = []


for i in tqdm(range(iterations)):
    rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration_order2(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U01, YA1, YB1)
    rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration_order2(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U02, YA2, YB2)
    rho_elmotA_T0 /= np.trace(rho_elmotA_T0)
    rho_elmotB_T0 /= np.trace(rho_elmotB_T0)
    rho_el_T0 /= np.trace(rho_el_T0)
    rho_T0_A /= np.trace(rho_T0_A)
    rho_T0_B /= np.trace(rho_T0_B)


    # IDEAL SPIN DENSITY MATRIX EVOLUTION
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)

    # Save info
    fidelity_alg.append(generalized_fidelity(rho_el_T0, rho_ideal))

np.savetxt(path + f'fidelity_alg_pulse_{temp_name}.txt', fidelity_alg, fmt='%.18f')