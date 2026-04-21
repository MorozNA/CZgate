import numpy as np
from tqdm import tqdm
# from src.algorithm.algorithm_tools_other import calculate_YrhoY


def one_iteration(rho_elmotA0, rho_elmotB0, rho_el0, rho_TA0, rho_TB0, U0, YA, YB):
    n = len(rho_TA0)

    rA = (YA @ (rho_elmotA0 - 1/2 * np.kron(rho_el0, rho_TA0)) @ YA.conj().T).reshape(9, n, 9, n)
    rB = (YB @ (rho_elmotB0 - 1/2 * np.kron(rho_el0, rho_TB0)) @ YB.conj().T).reshape(9, n, 9, n)

    r_elA = np.einsum('iaja->ij', rA)
    r_elB = np.einsum('iaja->ij', rB)
    rho_el1 = np.kron(U0, np.eye(n)) @ YB @ np.kron(r_elA, np.eye(n)) @ np.kron(np.eye(9), rho_TB0) @ YB.conj().T @ np.kron(U0.conj().T, np.eye(n))
    rho_el2 = np.kron(U0, np.eye(n)) @ YA @ np.kron(r_elB, np.eye(n)) @ np.kron(np.eye(9), rho_TA0) @ YA.conj().T @ np.kron(U0.conj().T, np.eye(n))
    rho_el = np.einsum('iaja->ij', rho_el1.reshape(9, n, 9, n)) + np.einsum('iaja->ij', rho_el2.reshape(9, n, 9, n))

    YrhoTB0Y = (YB @ np.kron(np.eye(9), rho_TB0) @ YB.conj().T).reshape(9, n, 9, n)
    YrhoTA0Y = (YA @ np.kron(np.eye(9), rho_TA0) @ YA.conj().T).reshape(9, n, 9, n)
    YrhoTB0Y_spin = np.einsum('iaja->ij', YrhoTB0Y)
    YrhoTA0Y_spin = np.einsum('iaja->ij', YrhoTA0Y)

    rho_TA = np.einsum('iajb,jk->iakb', rA, YrhoTB0Y_spin) + np.einsum('ij,jakb->iakb', r_elB, YrhoTA0Y)
    rho_TB = np.einsum('iajb,jk->iakb', rB, YrhoTA0Y_spin) + np.einsum('ij,jakb->iakb', r_elA, YrhoTB0Y)
    rho_TA = np.einsum('iaib->ab', rho_TA)
    rho_TB = np.einsum('iaib->ab', rho_TB)

    term1A_left_r = (np.kron(U0, np.eye(n)) @ YB).reshape(9, n, 9, n)
    term1A_right_r = (np.kron(np.eye(9), rho_TB0) @ YB.conj().T @ np.kron(U0.conj().T, np.eye(n))).reshape(9, n, 9, n)
    term1B_left_r = (np.kron(U0, np.eye(n)) @ YA).reshape(9, n, 9, n)
    term1B_right_r = (np.kron(np.eye(9), rho_TA0) @ YA.conj().T @ np.kron(U0.conj().T, np.eye(n))).reshape(9, n, 9, n)

    rho_elmotA = np.einsum(
        'iajb,jwkx,kbla->iwlx',
        term1A_left_r,
        rA,
        term1A_right_r,
        optimize=True
    )

    rho_elmotB = np.einsum(
        'iajb,jwkx,kbla->iwlx',
        term1B_left_r,
        rB,
        term1B_right_r,
        optimize=True
    )

    # rho_elmotA = np.zeros((9, n, 9, n), dtype=complex)
    # rho_elmotB = np.zeros((9, n, 9, n), dtype=complex)
    # for w1 in tqdm(range(n)):
    #     for w2 in range(n):
    #         rho_elmotA[:, w1, :, w2] = np.einsum('iajb,jk,kbla->il', term1A_left_r, rA[:, w1, :, w2], term1A_right_r)
    #         rho_elmotB[:, w1, :, w2] = np.einsum('iajb,jk,kbla->il', term1B_left_r, rB[:, w1, :, w2], term1B_right_r)
    rho_elmotA = rho_elmotA.reshape(9 * n, 9 * n) + rho_el2
    rho_elmotB = rho_elmotB.reshape(9 * n, 9 * n) + rho_el1


    return rho_elmotA, rho_elmotB, rho_el, rho_TA, rho_TB
