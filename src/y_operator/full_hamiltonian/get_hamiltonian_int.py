import numpy as np
from src.y_operator.consctust_spin_matrices import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator.construct_vib_matrices import get_V1_vib, get_V2_vib, get_W0z_vib, get_Wz_vib


def get_HA(om, tau, delta, xi, n, q):
    V1_spin = get_V1()
    V1_vib = get_V1_vib(n)

    V2_spin = get_V2()
    V2_vib = get_V2_vib(n, q)

    W0z_spin = get_W0z(2.0 * tau, om, tau, delta, xi)
    W0z_vib = get_W0z_vib(n)

    Wz_spin = get_Wz(2.0 * tau, om, tau, delta, xi)
    Wz_vib = get_Wz_vib(n)

    V1_full = np.kron(V1_spin, np.eye(3))
    V1_full = np.kron(V1_full, (np.kron(V1_vib, np.eye(n))))

    V2_full = np.kron(V2_spin, np.eye(3))
    V2_full = np.kron(V2_full, (np.kron(V2_vib, np.eye(n))))

    W0z_full = np.kron(W0z_spin, np.eye(3))
    W0z_full = np.kron(W0z_full, (np.kron(W0z_vib, np.eye(n))))

    Wz_full = np.kron(Wz_spin, np.eye(3))
    Wz_full = np.kron(Wz_full, (np.kron(Wz_vib, np.eye(n))))

    return V1_full + V2_full + W0z_full + Wz_full


def get_HB(om, tau, delta, xi, n, q):
    V1_spin = get_V1()
    V1_vib = get_V1_vib(n)

    V2_spin = get_V2()
    V2_vib = get_V2_vib(n, q)

    W0z_spin = get_W0z(2.0 * tau, om, tau, delta, xi)
    W0z_vib = get_W0z_vib(n)

    Wz_spin = get_Wz(2.0 * tau, om, tau, delta, xi)
    Wz_vib = get_Wz_vib(n)

    V1_full = np.kron(np.eye(3), V1_spin)
    V1_full = np.kron(V1_full, (np.kron(np.eye(n), V1_vib)))

    V2_full = np.kron(np.eye(3), V2_spin)
    V2_full = np.kron(V2_full, (np.kron(np.eye(n), V2_vib)))

    W0z_full = np.kron(np.eye(3), W0z_spin)
    W0z_full = np.kron(W0z_full, (np.kron(np.eye(n), W0z_vib)))

    Wz_full = np.kron(np.eye(3), Wz_spin)
    Wz_full = np.kron(Wz_full, (np.kron(np.eye(n), Wz_vib)))

    return V1_full + V2_full + W0z_full + Wz_full