import numpy as np
from scipy.linalg import sqrtm
from src.y_operator.constants import HBAR, kB
from src.y_operator.config import YOperatorDerived


def get_rho_T0(params: YOperatorDerived, T):
    n_check = 10000
    En_check = HBAR * params.OM_small * (np.arange(n_check) + 0.5)
    E_normalized_check = En_check / (kB * T)
    trace_check = np.sum(np.exp(-E_normalized_check))

    En = HBAR * params.OM_small * (np.arange(params.n) + 0.5)  # Energy levels
    E_normalized = En / (kB * T)
    rho_T0 = np.diag(np.exp(-E_normalized))
    rho_T0 = rho_T0 / trace_check
    return rho_T0 / np.trace(rho_T0)


def get_U0_ideal(tau, delta, xi):
    phi1 = delta * tau + xi + np.pi
    phi2 = 2 * phi1 - np.pi
    # phi2 = delta * tau
    return np.diag([1.0, np.exp(1j * phi1), 1.0, np.exp(1j * phi1), np.exp(1j * phi2), 1.0, 1.0, 1.0, 1.0])


def exact_evolution(rho_S0, U0):
    return U0 @ rho_S0 @ U0.conj().T


def generalized_fidelity(rho, sigma):
    # Compute sqrt(rho)
    sqrt_rho = sqrtm(rho)

    # Compute sqrt_rho * sigma * sqrt_rho
    intermediate = sqrt_rho @ sigma @ sqrt_rho

    # Take the matrix square root and compute the trace
    sqrt_intermediate = sqrtm(intermediate)
    fidelity = np.trace(sqrt_intermediate) ** 2

    # Ensure numerical stability (clip to [0, 1] due to floating point errors)
    return np.clip(fidelity.real, 0.0, 1.0)


from src.y_operator.construct_U0 import get_U, get_U_deltaR
from src.y_operator.change_basis import change_basis
def construct_U0_for_trotter(params, t):
    U1 = np.eye(9, dtype=complex)
    U1[1:3, 1:3] = get_U(params, t)
    U1[3:5, 3:5] = get_U(params, t)

    U_deltaR1 = get_U_deltaR(params, t)
    rows = [5, 6, 8]  # Target rows in U1
    cols = [5, 6, 8]  # Target columns in U1
    U1[np.ix_(rows, cols)] = U_deltaR1
    return change_basis(U1)
