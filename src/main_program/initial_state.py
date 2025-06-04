from src.y_operator.params import n, kB, HBAR, OM_small
import numpy as np


def get_rho_T0(T=1e-8):
    n_check = 5000
    En_check = HBAR * OM_small * (np.arange(n_check) + 0.5)
    E_normalized_check = En_check / (kB * T)
    trace_check = np.sum(np.exp(-E_normalized_check))

    En = HBAR * OM_small * (np.arange(n) + 0.5)  # Energy levels
    E_normalized = En / (kB * T)
    rho_T0 = np.diag(np.exp(-E_normalized))
    rho_T0 = rho_T0 / trace_check
    return rho_T0 / np.trace(rho_T0)


# hw / T -- gibbs
# avg = n -> 5mk -> 10 -> sqrt -> avg characteristic of linear scale
# делим это на ширину перетяжки и получаем безразмерный п-р вклада в фиделити


# rho_T0 = get_rho_T0(10e-6)
# print('initial_state.py:', np.trace(rho_T0))

# avg_numb = np.sqrt(2 * np.pi * 7.158e3)
# print()