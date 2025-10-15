import numpy as np
from src.y_operator.params import get_params
from src.y_operator.construct_U0 import get_U
from src.y_operator_deltaR.construct_U0 import get_U_deltaR


om = 2 * np.pi * 5e6
tau, delta, xi = get_params(om)
t = 2 * tau
deltaR = 5e12

U_no_leakage= get_U(delta, np.sqrt(2) * om, xi, t)
U_leakage = get_U_deltaR(delta, om, xi, t, deltaR)[:2, :2]


difference = np.linalg.norm(U_no_leakage - U_leakage)
print("Norm difference between U_3x3_reduced and U_2x2:", difference)