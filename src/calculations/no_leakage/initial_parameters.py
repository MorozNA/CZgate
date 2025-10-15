from src.y_operator.params import kB, HBAR, OM_small
import numpy as np
from src.y_operator.params import get_params
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B

delta_R = None
n = 50
OM = 2 * np.pi * 3.5e6
tau, delta, _ = get_params(OM)

print(2 * tau)

Q = 5e6
t_initial = 0.0
t_final = 2.0 * tau

Y_A = construct_Y_A(t_initial, t_final, OM, Q, n)
Y_B = construct_Y_B(t_initial, t_final, OM, Q, n)
U0 = construct_U0(t_final, OM)
U0_perfect = construct_U0(t_final, OM)


def get_rho_T0(T, n):
    n_check = 10000
    En_check = HBAR * OM_small * (np.arange(n_check) + 0.5)
    E_normalized_check = En_check / (kB * T)
    trace_check = np.sum(np.exp(-E_normalized_check))

    En = HBAR * OM_small * (np.arange(n) + 0.5)  # Energy levels
    E_normalized = En / (kB * T)
    rho_T0 = np.diag(np.exp(-E_normalized))
    rho_T0 = rho_T0 / trace_check
    return rho_T0 / np.trace(rho_T0)


temperature0 = 1e-9
temperature1 = 1e-6
temperature2 = 5e-6
temperature3 = 10e-6

rho_TA_0 = get_rho_T0(temperature0, n)
rho_TA_1 = get_rho_T0(temperature1, n)
rho_TA_2 = get_rho_T0(temperature2, n)
rho_TA_3 = get_rho_T0(temperature3, n)

rho_TB_0 = get_rho_T0(temperature0, n)
rho_TB_1 = get_rho_T0(temperature1, n)
rho_TB_2 = get_rho_T0(temperature2, n)
rho_TB_3 = get_rho_T0(temperature3, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1 / 2
rho_S0[4, 4] = 1 / 2
rho_S0[0, 4] = 1 / 2
rho_S0[4, 0] = 1 / 2
