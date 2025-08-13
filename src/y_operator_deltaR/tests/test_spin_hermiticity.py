from src.y_operator_deltaR.consctust_spin_matrices import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator_deltaR.params import get_params
import numpy as np
import pytest

# Initial parameters
om = 2 * np.pi * 3.5e6
tau, delta = get_params(om)
t_initial = 0.0
t_final = 2.0 * tau


def test_V1_hermiticity():
    """Test that V1 matrix is Hermitian at all times"""
    t_values = np.linspace(t_initial, t_final, 10)  # Test at multiple time points

    for t in t_values:
        V1 = get_V1(t, om)
        assert np.allclose(V1, V1.conj().T, atol=1e-10), \
            f"V1 is not Hermitian at t={t:.2f}"


def test_V2_hermiticity():
    """Test that V2 matrix is Hermitian at all times"""
    t_values = np.linspace(t_initial, t_final, 10)

    for t in t_values:
        V2 = get_V2(t, om)
        assert np.allclose(V2, V2.conj().T, atol=1e-10), \
            f"V2 is not Hermitian at t={t:.2f}"


def test_W0z_hermiticity():
    """Test that W0z matrix is Hermitian at all times"""
    t_values = np.linspace(t_initial, t_final, 10)

    for t in t_values:
        W0z = get_W0z(t, om)
        assert np.allclose(W0z, W0z.conj().T, atol=1e-10), \
            f"W0z is not Hermitian at t={t:.2f}"


def test_Wz_hermiticity():
    """Test that Wz matrix is Hermitian at all times"""
    t_values = np.linspace(t_initial, t_final, 10)

    for t in t_values:
        Wz = get_Wz(t, om)
        assert np.allclose(Wz, Wz.conj().T, atol=1e-10), \
            f"Wz is not Hermitian at t={t:.2f}"


def test_all_matrices_hermiticity():
    """Test all matrices at once with parameterized testing"""
    matrices = [
        ("V1", get_V1),
        ("V2", get_V2),
        ("W0z", get_W0z),
        ("Wz", get_Wz)
    ]

    t_values = np.linspace(t_initial, t_final, 5)

    for name, matrix_func in matrices:
        for t in t_values:
            mat = matrix_func(t, om)
            assert np.allclose(mat, mat.conj().T, atol=1e-10), \
                f"{name} is not Hermitian at t={t:.2f}"


if __name__ == "__main__":
    # Run individual tests
    test_V1_hermiticity()
    test_V2_hermiticity()
    test_W0z_hermiticity()
    test_Wz_hermiticity()
    print("All individual matrix hermiticity tests passed!")

    # Run comprehensive test
    test_all_matrices_hermiticity()
    print("All comprehensive hermiticity tests passed!")