import numpy as np
from calc_tools import get_U, calc_tau, calc_xi, calc_delta

delta_R = 1000
om = 1
delta = calc_delta(True, delta_R)
tau = calc_tau(delta, om, delta_R, True)
xi2 = calc_xi(delta, om, tau, True)
print(xi2 + np.pi)

state_01 = np.array([[1], [0]])
U1 = get_U(delta, om, 0.0, tau)
U2 = get_U(delta, om, xi2, tau)

intermidiate_state = U1 @ state_01
final_state = U2 @ intermidiate_state

print(np.round(intermidiate_state, 3))
print(np.round(final_state, 3))
v1 = np.round(np.linalg.norm(intermidiate_state), 3)
v2 = np.round(np.linalg.norm(final_state), 3)
print('Intermediate state norm: ', v1)
print('Final state norm: ', v2)
print(abs(final_state[0]))