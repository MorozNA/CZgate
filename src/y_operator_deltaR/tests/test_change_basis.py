import numpy as np
from src.y_operator_deltaR.construct_U0 import swap_basis
from src.y_operator_deltaR.construct_U0 import apply_hadamard


def swap_basis_to_check(matrix, i, j):
    matrix1 = np.copy(matrix)
    matrix1[[i, j], :] = matrix1[[j, i], :]  # Swap rows
    matrix1[:, [i, j]] = matrix1[:, [j, i]]  # Swap columns
    return matrix1


def apply_hadamard_to_check(matrix, i, j):
    # Define the Hadamard matrix
    Hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)

    matrix1 = matrix.copy()
    # Apply Hadamard to rows i and j
    submatrix = matrix1[[i, j], :].copy()  # Extract rows i and j
    matrix1[[i, j], :] = Hadamard @ submatrix  # Apply Hadamard to rows

    # Apply Hadamard to columns i and j
    submatrix = matrix1[:, [i, j]].copy()  # Extract columns i and j
    matrix1[:, [i, j]] = submatrix @ Hadamard.conj().T  # Apply Hadamard to columns

    return matrix1


def test_swap_basis():
    # Test cases: random matrices of different sizes
    test_sizes = [3, 5, 10, 50, 100]
    num_tests = 5

    for size in test_sizes:
        print(f"\nTesting matrix size: {size}x{size}")

        for _ in range(num_tests):
            # Generate a random matrix (real or complex)
            A = np.random.rand(size, size) + 1j * np.random.rand(size, size)
            i, j = np.random.randint(0, size, 2)  # Random indices to swap

            # Compute results using both methods
            A_swapped1 = swap_basis_to_check(A, i, j)
            A_swapped2 = swap_basis(A, i, j)

            # Check if results are identical (within floating-point tolerance)
            np.testing.assert_allclose(A_swapped1, A_swapped2, rtol=1e-10, atol=1e-10)

        print(f"✓ Passed {num_tests} random tests for size {size}x{size}")

    print("\nAll tests passed! Both functions produce identical results.")


def test_hadamard():
    """Test if both methods produce identical results."""
    np.random.seed(0)  # For reproducibility
    sizes = [3, 5, 10, 50]  # Test different matrix sizes
    num_tests = 5

    for size in sizes:
        print(f"\nTesting matrix size {size}x{size}:")
        for _ in range(num_tests):
            # Generate random unitary matrix (complex)
            matrix = np.random.rand(size, size) + 1j * np.random.rand(size, size)
            i, j = np.random.choice(size, 2, replace=False)  # Random distinct indices

            # Apply both methods
            result1 = apply_hadamard(matrix.copy(), i, j)
            result2 = apply_hadamard_to_check(matrix.copy(), i, j)

            # Check if results are numerically equivalent
            np.testing.assert_allclose(result1, result2, atol=1e-10, err_msg=f"Failed for size {size}, indices ({i}, {j})")

        print(f"✓ Passed {num_tests} tests for size {size}x{size}")