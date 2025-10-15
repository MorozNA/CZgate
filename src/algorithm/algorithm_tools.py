import numpy as np


def calculate_rho_wprx_wx(rho_S0, rho_T0X, Y_X, wprx, wx):
    n = len(rho_T0X)
    " Trace is taken in algorithm program "
    Y_X_reshaped = Y_X.reshape(9, n, 9, n)  # np.kron(a_ij, b_kl) + reshape(i,k,j,l) = ij,kl->ikjl
    rho_w = np.zeros((9, 9), dtype=complex)
    for vprx in range(n):
        for vx in range(n):
            term = Y_X_reshaped[:, wprx, :, vprx] @ (rho_S0 * rho_T0X[vprx, vx]) @ Y_X_reshaped[:, wx, :, vx].conj().T
            rho_w += term
    return rho_w


def calculate_rho_SX(rho_S0, rho_T0X, Y_X):
    # calculate either rho_SA or rho_SB
    rho_SX = np.zeros((9, 9), dtype=complex)
    for wx in range(len(rho_T0X)):
        rho_SX += calculate_rho_wprx_wx(rho_S0, rho_T0X, Y_X, wx, wx)
    return rho_SX


def calculate_spin_density_withX(rho_SX, rho_T0notX, U0, Y_notX):
    n = len(rho_T0notX)

    """
    1) calculate rho_SB
    2) Use rho_S formula
    """

    # Reshape Y_A to (9, n, 9, n) for easier indexing
    Y_notX_reshaped = Y_notX.reshape(9, n, 9, n)

    # Initialize the result matrix
    rho_S = np.zeros((9, 9), dtype=complex)

    # Loop over all possible v_A, w_A combinations
    for w_notX in  range(n):
        for vpr_notX in range(n):
            for v_notX in range(n):
                # Calculate the term inside the sum
                term = U0 @ Y_notX_reshaped[:, w_notX, :, vpr_notX] @ (rho_SX * rho_T0notX[vpr_notX, v_notX]) @ \
                            Y_notX_reshaped[:, w_notX, :, v_notX].conj().T @ U0.conj().T
                rho_S += term
    return rho_S
