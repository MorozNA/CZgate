from src.y_operator.consctust_spin_matrices import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.params import get_params
import numpy as np

# Initial parameters
om = 2 * np.pi * 3.5e6
tau, delta = get_params(om)
t_initial = 0.0
t_final = 2.0 * tau


def test_U0_unitarity():
    """Test that U0 is unitary (U0† U0 = I) at all times"""
    t_values = np.linspace(t_initial, t_final, 200)  # Test at multiple time points
    U0 = construct_U0(t_values[0], om)
    identity = np.eye(U0.shape[0])  # Assuming U0

    for t in t_values:
        U0 = construct_U0(t, om)
        # Check both conditions for complete unitarity
        product1 = U0 @ U0.conj().T
        product2 = U0.conj().T @ U0

        assert np.allclose(product1, identity, atol=1e-10), \
            f"U0(t)U0†(t) ≠ I at t={t:.4f}"
        assert np.allclose(product2, identity, atol=1e-10), \
            f"U0†(t)U0(t) ≠ I at t={t:.4f}"


def test_matrix_fun_hermiticity_A():
    """Test that integrand in get_integrand_A is Hermitian for all matrix functions"""
    matrix_functions = [get_V1, get_V2, get_W0z, get_Wz]
    t_values = np.linspace(t_initial, t_final, 200)  # Test at multiple time points

    for matrix_fun in matrix_functions:
        for t in t_values:
            matrix_fun_value = np.kron(matrix_fun(t, om), np.eye(3))
            # Check Hermiticity
            assert np.allclose(matrix_fun_value, matrix_fun_value.conj().T, atol=1e-10), \
                f"Integrand for {matrix_fun.__name__} in A configuration not Hermitian at t={t:.4f}"


def test_integrand_hermiticity_A():
    """Test that integrand in get_integrand_A is Hermitian for all matrix functions"""
    matrix_functions = [get_V1, get_V2, get_W0z, get_Wz]
    t_values = np.linspace(t_initial, t_final, 200)  # Test at multiple time points

    for matrix_fun in matrix_functions:
        for t in t_values:
            U0 = construct_U0(t, om)
            matrix_value = matrix_fun(t, om)#  * (HBAR * M)
            matrix_value = np.kron(matrix_value, np.eye(3))
            integrand = U0 @ matrix_value @ U0.conj().T
            # Check Hermiticity
            assert np.allclose(integrand, integrand.conj().T, atol=1e-10), \
                f"Integrand for {matrix_fun.__name__} in A configuration not Hermitian at t={t:.4f}"


def test_integrand_hermiticity_B():
    """Test that integrand in get_integrand_B is Hermitian for all matrix functions"""
    matrix_functions = [get_V1, get_V2, get_W0z, get_Wz]
    t_values = np.linspace(t_initial, t_final, 200)

    for matrix_fun in matrix_functions:
        for t in t_values:
            U0 = construct_U0(t, om)
            integrand = U0 @ (np.kron(np.eye(3), matrix_fun(t, om))) @ U0.conj().T
            # Check Hermiticity
            assert np.allclose(integrand, integrand.conj().T, atol=1e-10), \
                f"Integrand for {matrix_fun.__name__} in B configuration not Hermitian at t={t:.4f}"


def test_symmetrized_integrand_hermiticity():
    """Test that the symmetrized integrand (0.5*(M + M†)) remains Hermitian"""
    matrix_functions = [get_V1, get_V2, get_W0z, get_Wz]
    t_values = np.linspace(t_initial, t_final, 200)

    for matrix_fun in matrix_functions:
        for t in t_values:
            # Test configuration A
            U0 = construct_U0(t, om)
            M = U0 @ (np.kron(matrix_fun(t, om), np.eye(3))) @ U0.conj().T
            symmetrized = 0.5 * (M + M.conj().T)
            assert np.allclose(symmetrized, symmetrized.conj().T, atol=1e-10), \
                f"Symmetrized integrand A for {matrix_fun.__name__} not Hermitian at t={t:.4f}"

            # Test configuration B
            M = U0 @ (np.kron(np.eye(3), matrix_fun(t, om))) @ U0.conj().T
            symmetrized = 0.5 * (M + M.conj().T)
            assert np.allclose(symmetrized, symmetrized.conj().T, atol=1e-10), \
                f"Symmetrized integrand B for {matrix_fun.__name__} not Hermitian at t={t:.4f}"


if __name__ == "__main__":
    test_integrand_hermiticity_A()
    test_integrand_hermiticity_B()
    test_symmetrized_integrand_hermiticity()
    print("All integrand hermiticity tests passed!")