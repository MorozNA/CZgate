import numpy as np
from scipy.linalg import sqrtm


def generalized_fidelity(rho, sigma):
    """
    Computes the generalized fidelity between two density matrices rho and sigma.

    Parameters:
    -----------
    rho : np.ndarray
        First density matrix (positive semi-definite, trace 1).
    sigma : np.ndarray
        Second density matrix (positive semi-definite, trace 1).

    Returns:
    --------
    fidelity : float
        Generalized fidelity F(rho, sigma) in [0, 1].
    """
    # Compute sqrt(rho)
    sqrt_rho = sqrtm(rho)

    # Compute sqrt_rho * sigma * sqrt_rho
    intermediate = sqrt_rho @ sigma @ sqrt_rho

    # Take the matrix square root and compute the trace
    sqrt_intermediate = sqrtm(intermediate)
    fidelity = np.trace(sqrt_intermediate) ** 2

    # Ensure numerical stability (clip to [0, 1] due to floating point errors)
    return np.clip(fidelity.real, 0.0, 1.0)


def concurrence(rho_AB, dim_A, dim_B):
    tensor = rho_AB.reshape((dim_A, dim_B, dim_A, dim_B))
    rho_A = np.einsum('ikjk->ij', tensor)
    rho_B = np.einsum('ikil->kl', tensor)
    return np.sqrt(np.real(1 + np.trace(rho_AB @ rho_AB) - np.trace(rho_A @ rho_A) - np.trace(rho_B @ rho_B)))


def negativity(rho, dim_A, dim_B):
    """
    Compute the negativity of a bipartite density matrix.

    Parameters:
        rho (np.ndarray): Density matrix (shape (dim_A * dim_B, dim_A * dim_B))
        dim_A (int): Dimension of subsystem A
        dim_B (int): Dimension of subsystem B

    Returns:
        float: Negativity of the state
    """
    # Reshape the density matrix into a tensor for partial transposition
    rho_reshaped = rho.reshape((dim_A, dim_B, dim_A, dim_B))

    # Perform partial transpose on subsystem B (swap indices 1 and 3)
    rho_partial_transpose = np.transpose(rho_reshaped, (0, 3, 2, 1))

    # Reshape back to a matrix
    rho_partial_transpose = rho_partial_transpose.reshape((dim_A * dim_B, dim_A * dim_B))

    # Compute eigenvalues of the partially transposed matrix
    eigenvalues = np.linalg.eigvals(rho_partial_transpose)

    # Sum the absolute values of negative eigenvalues
    neg = np.sum(np.abs(eigenvalues[eigenvalues < 0]))

    return neg


def logarithmic_negativity(rho, dim_A, dim_B):
    rho_pt = np.transpose(rho.reshape((dim_A, dim_B, dim_A, dim_B)), (0, 3, 2, 1)).reshape((dim_A*dim_B, dim_A*dim_B))
    return np.log2(np.sum(np.abs(np.linalg.eigvals(rho_pt))))


def realignment_criterion(rho, dim_A, dim_B):
    """Compute realignment criterion (returns ||R(ρ)||₁)."""
    rho_realigned = rho.reshape((dim_A, dim_B, dim_A, dim_B)).transpose(0, 2, 1, 3).reshape((dim_A**2, dim_B**2))
    return np.sum(np.linalg.svd(rho_realigned, compute_uv=False))