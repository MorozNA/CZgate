import numpy as np
n = 100


def get_rho_s(rho_S, rho_T, Y):
    # проверить сопрягается матрица или матричный элемент
    rho = Y @ np.kron(rho_S, rho_T) @ Y.conj().T
    rho = np.einsum('iaja', rho.reshape(9, n, 9, n))
    return rho


def get_rho_vib(rho_S, rho_T, Y):
    rho = Y @ np.kron(rho_S, rho_T) @ Y.conj().T
    rho = np.einsum('iaib', rho.reshape(9, n, 9, n))
    return rho