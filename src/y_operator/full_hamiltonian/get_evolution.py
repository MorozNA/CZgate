import numpy as np
from scipy.linalg import expm
from src.y_operator.config import YOperatorDerived
from src.y_operator.full_hamiltonian.get_hamiltonian0 import get_H0
from src.y_operator.full_hamiltonian.get_hamiltonian_int import get_HA, get_HB
from dataclasses import replace


def get_evolution(params: YOperatorDerived):
    p_xi0_om = replace(params, xi=0.0)
    H01 = np.kron(get_H0(p_xi0_om), np.kron(np.eye(params.n), np.eye(params.n)))
    H02 = np.kron(get_H0(params), np.kron(np.eye(params.n), np.eye(params.n)))
    H1 = H01 + get_HA(p_xi0_om) + get_HB(p_xi0_om)
    H2 = H02 + get_HA(params) + get_HB(params)
    return expm(-1j * H2 * params.tau) @ expm(-1j * H1 * params.tau)

