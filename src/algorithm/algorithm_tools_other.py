import numpy as np


def calculate_YrhoY(rho_S0, rho_T0X, Y_X):
    n = len(rho_T0X)
    # YrhoY = Y_X @ np.kron(rho_S0, rho_T0X) @ Y_X.conj().T
    " Trace is taken in algorithm program "
    Y_X_reshaped = Y_X.reshape(9, n, 9, n)  # np.kron(a_ij, b_kl) + reshape(i,k,j,l) = ij,kl->ikjl
    YrhoY_tensor = np.einsum(
        'iajb,jbkc,kcld->iald',
        Y_X_reshaped,
        np.kron(rho_S0, rho_T0X).reshape(9, n, 9, n),
        Y_X_reshaped.conj().transpose(2, 3, 0, 1),
        optimize=True
    )
    return YrhoY_tensor
