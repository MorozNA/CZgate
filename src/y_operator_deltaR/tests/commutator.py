# from scipy.sparse import kron, identity, csr_matrix
# def construct_Y(t_initial, t_final):
#     Y = construct_Y_A(t_initial, t_final).reshape(9, n, 9, n)
#     I_n = identity(n, format='csr')
#     Y_A = kron(csr_matrix(Y.transpose([1, 0, 3, 2]).reshape(9 * n, 9 * n)), I_n)
#     Y_B = kron(I_n, csr_matrix(Y.reshape(9 * n, 9 * n)))
#     return Y_A, Y_B
