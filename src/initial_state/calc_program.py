import numpy as np
from initial_state import rho_T0
from evolution_fun import get_rho_s, get_rho_vib
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.construct_Y import construct_Y_A
from src.y_operator.params import tau


t_initial = 0.0
t_final = 2.0 * tau
Y_A = construct_Y_A(t_initial, t_final)
print(np.round(Y_A, 2))

U0 = construct_U0(t_final)
rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1/2
rho_S0[4, 4] = 1/2
rho_S0[0, 4] = 1/2
rho_S0[4, 0] = 1/2
print(Y_A.shape, rho_S0.shape, rho_T0.shape)

# rho_vib_A = get_rho_vib(rho_S0, rho_T0, Y_A)
# rho_SA = get_rho_s(rho_S0, rho_T0, Y_A)

rho_S = get_rho_s(rho_S0, rho_T0, Y_A)
rho_S = U0 @ rho_S @ U0.conj().T
print(np.round(rho_S, 3))

print('\n')
rho_expected = U0 @ rho_S0 @ U0.conj().T
print(np.round(rho_expected, 3))

print(np.trace(rho_S @ rho_expected))
