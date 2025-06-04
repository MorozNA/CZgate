import numpy as np
from src.y_operator.params import n


def get_rho_T_partial(rho_S, rho_T, Y):
    Y = Y.reshape(9, n, 9, n)
    rho_wb = np.zeros((n, 9, 9), dtype=complex)
    for i in range(n):
        for j in range(n):
            Y_conj = Y[:, i, :, j].conj().T
            rho_wb[i] += Y[:, i, :, j] @ (rho_S * rho_T[j, j]) @ Y_conj
    return rho_wb


def get_rho_S_partial(rho_S, rho_T, Y):
    rho = np.sum(get_rho_T_partial(rho_S, rho_T, Y), axis=0)
    return rho / np.trace(rho)  # TODO: check what happens with the trace


def get_rho_s(rho_S, rho_T, Y, U0):
    Y = Y.reshape(9, n, 9, n)
    for i in range(n):
        for j in range(n):
            Y_conj = Y[:, i, :, j].conj().T
            rho_S += Y[:, i, :, j] @ rho_S * (rho_T[j, j]) @ Y_conj
    # проверить сопрягается матрица или матричный элемент
    # rho = Y @ np.kron(rho_S, rho_T) @ Y.conj().T
    # rho = np.einsum('iaja', rho.reshape(9, n, 9, n))
    return U0 @ rho_S @ U0.conj().T


# def get_rho_vib(rho_S, rho_T, Y):
#     rho = Y @ np.kron(rho_S, rho_T) @ Y.conj().T
#     rho = np.einsum('iaib', rho.reshape(9, n, 9, n))
#     return rho