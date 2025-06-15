import numpy as np
from src.y_operator.params import get_params
from src.y_operator.construct_U0 import get_U, change_basis


def construct_U0_elimination(t, om, delta_rydberg=None):
    tau, delta, xi = get_params(om, delta_rydberg)
    if t <= tau:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, om, 0.0, t)
        U1[3:5, 3:5] = get_U(delta, om, 0.0, t)
        if delta_rydberg is not None:
            ## :TODO: delta - om**2 / 2 / delta_R or delta - (np.sqrt * om) ** 2 / 2 / delta_R
            U1[5:7, 5:7] = get_U(delta - (np.sqrt(2) * om) ** 2 / 2 / delta_rydberg, np.sqrt(2) * om, 0.0, t)
        else:
            U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, 0.0, t)
        return change_basis(U1)
    else:
        U1 = np.eye(9, dtype=complex)
        U1[1:3, 1:3] = get_U(delta, om, 0.0, tau)
        U1[3:5, 3:5] = get_U(delta, om, 0.0, tau)
        if delta_rydberg is not None:
            U1[5:7, 5:7] = get_U(delta - (np.sqrt(2) * om) ** 2 / 2 / delta_rydberg, np.sqrt(2) * om, 0.0, tau)
        else:
            U1[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, 0.0, tau)
        U2 = np.eye(9, dtype=complex)
        U2[1:3, 1:3] = get_U(delta, om, xi, t - tau)
        U2[3:5, 3:5] = get_U(delta, om, xi, t - tau)
        if delta_rydberg is not None:
            U2[5:7, 5:7] = get_U(delta - (np.sqrt(2) * om) ** 2 / 2 / delta_rydberg, np.sqrt(2) * om, xi, t - tau)
        else:
            U2[5:7, 5:7] = get_U(delta, np.sqrt(2) * om, xi, t - tau)
        if not np.allclose((U2 @ U1) @ (U2 @ U1).conj().T, np.eye(U2.shape[0])):
            raise ValueError("Input matrix U2 is not unitary!")
        return change_basis(U2 @ U1)