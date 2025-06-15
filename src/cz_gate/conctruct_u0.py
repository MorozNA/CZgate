import numpy as np
from calc_tools import get_U, calc_tau, calc_xi, calc_delta


def conctruct_U0(t):
    om = 1
    delta = calc_delta()
    tau = calc_tau(delta, om)
    xi2 = calc_xi(delta, om, tau)

    U0 = np.zeros((9, 9), dtype=complex)

    if t <= tau:
        t1 = t
    else:
        t1 = tau

    U0_1block = 1
    U0_2block = get_U(delta, om, 0.0, t1)
    U0_3block = U0_2block
    U0_4block = get_U(delta, np.sqrt(2) * om, 0.0, t1)
    U0_5block = np.eye(2, dtype=complex)

    U0[0, 0] = U0_1block
    U0[1:3, 1:3] = U0_2block
    U0[3:5, 3:5] = U0_3block
    U0[5:7, 5:7] = U0_4block
    U0[7:9, 7:9] = U0_5block

    if (t > tau) & (t < 2 * tau):
        t2 = t - tau
        U2 = np.zeros((9, 9), dtype=complex)

        U2_1block = 1
        U2_2block = get_U(delta, om, xi2, t2)
        U2_3block = U2_2block
        U2_4block = get_U(delta, np.sqrt(2) * om, xi2, t2)
        U2_5block = np.eye(2, dtype=complex)

        U2[0, 0] = U2_1block
        U2[1:3, 1:3] = U2_2block
        U2[3:5, 3:5] = U2_3block
        U2[5:7, 5:7] = U2_4block
        U2[7:9, 7:9] = U2_5block

        U0 = U2 @ U0

    return U0
