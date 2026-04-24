import numpy as np
from src.y_operator.calc_params import calc_params, get_delta_renorm


om = 2 * np.pi * 5e6
deltaR_array = np.linspace(50, 50000, 10) * 2 * np.pi * 1e6

for deltaR in deltaR_array:
    tau, delta, xi = calc_params(om, None)
    tau_leakage, delta_leakage, xi_leakage = calc_params(om, deltaR)
    print('______________________ \n')
    print('delta_R - 2 delta = ', (deltaR - 2 * delta) / 1e6, '(MHz)')
    print('delta (renorm) diff = ', (get_delta_renorm(delta_leakage, om, deltaR) - delta_leakage) / 1e6, '(MHz)')
    print('\n')
    print('delta diff = ', (delta - delta_leakage) / 1e6, '(MHz)')
    print('tau diff = ', (tau - tau_leakage) * 1e9, '(ns)')
    print('xi diff = ', xi - xi_leakage)
    print('______________________ \n')