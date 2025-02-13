import numpy as np
from src.y_operator.params import tau
from src.y_operator.construct_Y import construct_Y_A

t_initial = 0.0
t_final = 2.0 * tau
Y_A = construct_Y_A(t_initial, t_final)


commut = Y_A @ Y_B - Y_B @ Y_A
print(np.amax(abs(Y_A)))
print(np.amax(abs(Y_B)))
print(np.amax(abs(commut)))
print(np.amax(abs(Y_A - Y_B)))