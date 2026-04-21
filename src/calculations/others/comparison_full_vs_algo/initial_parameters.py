import numpy as np
from src.algorithm.other_tools import get_rho_T0
from src.y_operator_deltaR.params import lambd_1, lambd_2, get_params
from src.y_operator_deltaR.full_hamiltonian.get_evolution import get_evolution
from src.y_operator_deltaR.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator_deltaR.construct_U0 import construct_U0


temperature0 = 1e-9
n = 15
# Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
Q = 0.0
om = 2 * np.pi * 5e6
delta_R = 2 * np.pi * 50e6
tau, delta, xi = get_params(om, delta_R)

rho_TA_0 = get_rho_T0(temperature0, n)
rho_TB_0 = get_rho_T0(temperature0, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1 / 2
rho_S0[4, 4] = 1 / 2
rho_S0[0, 4] = 1 / 2
rho_S0[4, 0] = 1 / 2


### GET EVOLUTION OPERATORS
t_initial = 0.0
t_final = 2 * tau

U0 = construct_U0(t_final, om, tau, delta, xi, delta_R)
YA = construct_Y_A(t_initial, t_final, om, tau, delta, xi, delta_R, Q, n)
YB = construct_Y_B(t_initial, t_final, om, tau, delta, xi, delta_R, Q, n)

# diag_Y_A = np.diag(np.diag(YA))
# print(np.allclose(YA, diag_Y_A))
# print(np.sum(YA) - np.sum(np.diag(diag_Y_A)))

# big_U0 = np.kron(U0, np.kron(np.eye(n), np.eye(n)))
# big_YA = np.kron(YA, np.eye(n))
# big_YB = np.kron(np.eye(n), YB)
# big_evol = big_U0 @ big_YA @ big_YB


full_U = get_evolution(delta, om, xi, delta_R, tau, n, Q)
# print(np.allclose(full_U, big_evol))


### SET UP INITIAL STATES
exact_rho = np.copy(rho_S0)
calc_rho = np.copy(rho_S0)
rho_TA = np.copy(rho_TA_0)
rho_TB = np.copy(rho_TB_0)

rho_full = np.kron(rho_S0, np.kron(rho_TA_0, rho_TB_0))