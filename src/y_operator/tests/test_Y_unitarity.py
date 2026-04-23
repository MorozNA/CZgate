from src.y_operator.construct_Y import construct_Y_A, construct_Y_B, get_integral_atom_A, get_integral_atom_B
from src.y_operator.construct_Y import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator.construct_Y import get_V1_vib, get_V2_vib, get_W0z_vib, get_Wz_vib
from src.y_operator.construct_Y import integrate_matrix_A, integrate_matrix_B
from src.y_operator.params import get_params
import numpy as np


def test_hermiticity_integrated_matrices_A():
    """Test that all integrated matrices in construct_Y_A are Hermitian"""
    om = 2 * np.pi * 3.5e6
    tau, delta = get_params(om)
    t_initial = 0.0
    t_final = 2.0 * tau

    # Test each matrix function
    print('\n')
    for matrix_fun in [get_V1, get_V2, get_W0z, get_Wz]:
        integrated_matrix = integrate_matrix_A(t_initial, t_final, om, matrix_fun)
        print(np.amax(abs(integrated_matrix)))
        assert np.allclose(integrated_matrix, integrated_matrix.conj().T, atol=1e-6), \
            f"Integrated matrix for {matrix_fun.__name__} in A is not Hermitian"


def test_hermiticity_integrated_matrices_B():
    """Test that all integrated matrices in construct_Y_B are Hermitian"""
    om = 2 * np.pi * 3.5e6
    tau, delta = get_params(om)
    t_initial = 0.0
    t_final = 2.0 * tau

    # Test each matrix function
    for matrix_fun in [get_V1, get_V2, get_W0z, get_Wz]:
        integrated_matrix = integrate_matrix_B(t_initial, t_final, om, matrix_fun)
        assert np.allclose(integrated_matrix, integrated_matrix.conj().T, atol=1e-6), \
            f"Integrated matrix for {matrix_fun.__name__} in B is not Hermitian"


def test_hermiticity_vibrational_matrices():
    n = 100
    q = 5e6
    for matrix in [get_V1_vib(n), get_V2_vib(n, q), get_W0z_vib(n), get_Wz_vib(n)]:
        assert np.allclose(matrix, matrix.conj().T, atol=1e-6), \
            f"Vibrational matrix for {matrix.__name__} is not Hermitian"


def test_final_integral_fun():
    n = 100
    om = 2 * np.pi * 3.5e6
    tau, delta = get_params(om)
    t_initial = 0.0
    t_final = 2.0 * tau
    q = 5e6

    integral_A = get_integral_atom_A(t_initial, t_final, om, q, n)
    integral_B = get_integral_atom_B(t_initial, t_final, om, q, n)

    assert np.allclose(integral_A, integral_A.conj().T, atol=1e-8), \
        "Combined integral in A is not Hermitian"
    assert np.allclose(integral_B, integral_B.conj().T, atol=1e-8), \
        "Combined integral in A is not Hermitian"


def test_Y_unitarity():
    n = 100
    om = 2 * np.pi * 3.5e6
    tau, delta = get_params(om)
    t_initial = 0.0
    t_final = 2.0 * tau
    q = 5e6

    Y_A = construct_Y_A(t_initial, t_final, om, q, n)
    Y_B = construct_Y_B(t_initial, t_final, om, q, n)

    # Verify unitarity
    identity = np.eye(Y_A.shape[0])
    product_A1 = Y_A @ Y_A.conj().T
    product_A2 = Y_A.conj().T @ Y_A

    product_B1 = Y_B @ Y_B.conj().T
    product_B2 = Y_B.conj().T @ Y_B

    # diff_A1 = np.max(np.abs(product_A1 - identity))
    # diff_A2 = np.max(np.abs(product_A2 - identity))
    # diff_B1 = np.max(np.abs(product_B1 - identity))
    # diff_B2 = np.max(np.abs(product_B2 - identity))

    assert np.allclose(product_A1, identity, atol=1e-8), \
        f"exp(-iH) not unitary (UU† test) for case 1"
    assert np.allclose(product_A2, identity, atol=1e-8), \
        f"exp(-iH) not unitary (U†U test) for case 2"
    assert np.allclose(product_B1, identity, atol=1e-8), \
        f"exp(-iH) not unitary (UU† test) for case 1"
    assert np.allclose(product_B2, identity, atol=1e-8), \
        f"exp(-iH) not unitary (U†U test) for case 2"


if __name__ == "__main__":
    # Run tests
    test_hermiticity_integrated_matrices_A()
    test_hermiticity_integrated_matrices_B()
    test_hermiticity_vibrational_matrices()

    test_final_integral_fun()
    test_Y_unitarity()
    print("All tests passed!")
