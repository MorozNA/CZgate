import numpy as np
import random
from scipy.linalg import expm


def get_U(delta, omega, xi, t):
    om_0 = np.sqrt(delta ** 2 + omega ** 2)
    cos_term = np.cos(om_0 * t / 2)
    sin_term = np.sin(om_0 * t / 2)
    U_bb = cos_term - 1j * delta / om_0 * sin_term
    U_br = -1j * (omega * np.exp(1j * xi)) / om_0 * sin_term
    U_rb = -1j * (omega * np.exp(-1j * xi)) / om_0 * sin_term
    U_rr = cos_term + 1j * delta / om_0 * sin_term

    U = np.array([[U_bb, U_br], [U_rb, U_rr]], dtype=complex) * np.exp(1j * delta * t / 2)
    return U


def get_U_hamilton(delta, om, xi, t):
    Hamiltonian = np.zeros((2, 2), dtype=complex)
    Hamiltonian[0, 0] = 0.0
    Hamiltonian[0, 1] = om / 2 * np.exp(1j * xi)
    Hamiltonian[1, 0] = Hamiltonian[0, 1].conj()
    Hamiltonian[1, 1] = -delta
    evol_U = expm(-1j * Hamiltonian * t)
    if not np.allclose(evol_U @ evol_U.conj().T, np.eye(evol_U.shape[0])):
        raise ValueError("Output matrix U_delta is not unitary!")
    return evol_U


def test_matrix_equivalence():
    """Test that get_U and get_U_hamilton produce the same unitary matrix."""
    # Test parameters
    test_params = [
        (0.377, 1.0, 0.86, 4.29),  # original test case
        # Add these new test cases:
        (0.1, 0.5, 0.1, 1.0),  # small values
        (1.0, 1.0, 1.0, 1.0),  # all equal
        (0.0, 1.0, 0.5, 2.0),  # delta = 0
        (0.5, 0.0, 0.3, 1.5),  # omega = 0
        (0.2, 0.8, 0.0, 3.0),  # xi = 0
        (0.3, 0.7, 0.4, 0.0),  # t = 0
        (10.0, 5.0, 2.0, 3.0),  # larger values
        (0.001, 0.002, 0.003, 0.004),  # very small values
        (1.57, 3.14, 6.28, 1.57),  # pi-related values
        (1.0, 10.0, 0.1, 10.0),  # large omega and t
        (0.5, 1.0, 1.0, 100.0),  # large time
        (1e-5, 1e-5, 1e-5, 1e-5),  # very small all
        # (1e5, 1e5, 1e5, 1e5),  # very large all
    ]
    for _ in range(5):  # add 5 random test cases
        test_params.append(tuple(10 ** random.uniform(-5, 5) for _ in range(4)))

    for delta, omega, xi, t in test_params:
        # Calculate matrices using both methods
        U_direct = get_U(delta, omega, xi, t)
        U_hamilton = get_U_hamilton(delta, omega, xi, t)

        print('\n')
        print(np.round(U_direct, 2), '\n')
        print(np.round(U_hamilton, 2), '\n')

        # Check if they're close (within numerical precision)
        assert np.allclose(U_direct, U_hamilton), \
            f"Matrices differ for params delta={delta}, omega={omega}, xi={xi}, t={t}"

