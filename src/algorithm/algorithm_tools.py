import numpy as np


def calculate_YrhoY(rho_S0, rho_T0X, Y_X):
    n = len(rho_T0X)
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


def calculate_rho_wprx_wx(YrhoY_tensor, wprx, wx):
    return YrhoY_tensor[:, wprx, : wx]


def calculate_rho_SX(YrhoY_tensor):
    return np.einsum('iaja->ij', YrhoY_tensor)


def calculate_spin_density_withX(rho_SX, rho_T0notX, U0, Y_notX):
    n = len(rho_T0notX)

    """
    1) calculate rho_SB
    2) Use rho_S formula
    """

    # Reshape Y_A to (9, n, 9, n) for easier indexing
    Y_notX_reshaped = Y_notX.reshape(9, n, 9, n)

    # Initialize the result matrix
    inner_tensor = np.einsum(
        'iajb,jbkc,kcla->il',
        Y_notX_reshaped,
        np.kron(rho_SX, rho_T0notX).reshape(9, n, 9, n),
        Y_notX_reshaped.conj().transpose(2, 3, 0, 1),
        optimize=True
    )
    return U0 @ inner_tensor @ U0.conj().T