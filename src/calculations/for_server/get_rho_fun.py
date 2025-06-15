from src.y_operator.params import kB, HBAR, OM_small
import numpy as np

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


# print(np.trace(get_rho_T0(1e-9, 100)))
# print(np.trace(get_rho_T0(1e-6, 100)))
# print(np.trace(get_rho_T0(3e-6, 100)))
# print(np.trace(get_rho_T0(5e-6, 150)))