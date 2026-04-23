import numpy as np
import random
from scipy.linalg import expm
from src.y_operator.construct_U0 import get_U
from src.y_operator.full_hamiltonian.get_hamiltonian0 import get_H_simple, get_H0


def get_U_hamilton(delta, om, xi, t):
    evol_U = expm(-1j * get_H_simple(delta, om, xi) * t)
    if not np.allclose(evol_U @ evol_U.conj().T, np.eye(evol_U.shape[0])):
        raise ValueError("Output matrix U_delta is not unitary!")
    return evol_U


def test_simple_hamiltonian_equivalence():
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

