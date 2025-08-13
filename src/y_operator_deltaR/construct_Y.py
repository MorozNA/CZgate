import numpy as np
from scipy.linalg import expm
from src.y_operator_deltaR.get_integrals import get_integral_atom_A, get_integral_atom_B


def construct_Y_A(t_initial, t_final, om, tau, delta, xi, delta_rydberg, q, n):
    integral_atomA = get_integral_atom_A(t_initial, t_final, om, tau, delta, xi, delta_rydberg, q, n)
    Y_A = expm(-1j * integral_atomA)  # division by HBAR is already accounted for in matrix definition
    return Y_A


def construct_Y_B(t_initial, t_final, om, tau, delta, xi, delta_rydberg, q, n):
    integral_atomB = get_integral_atom_B(t_initial, t_final, om, tau, delta, xi, delta_rydberg, q, n)
    Y_B = expm(-1j * integral_atomB)  # division by HBAR is already accounted for in matrix definition
    return Y_B
