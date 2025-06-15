import numpy as np
from src.calculations.for_server.get_rho_fun import get_rho_T0
from src.y_operator.params import lambd_1, lambd_2


temperature0 = 5e-6
n = 100
Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)


rho_TA_0 = get_rho_T0(temperature0, n)
rho_TB_0 = get_rho_T0(temperature0, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1 / 2
rho_S0[4, 4] = 1 / 2
rho_S0[0, 4] = 1 / 2
rho_S0[4, 0] = 1 / 2
