from scipy.linalg import expm
import numpy as np
from src.y_operator_deltaR.construct_Y import integrate_matrix_A, integrate_matrix_B
from src.y_operator_deltaR.construct_Y import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator_deltaR.construct_Y import V1_vib, get_V2_vib, W0z_vib, Wz_vib
from src.y_operator_deltaR.params import get_params

"""Test that the final combined integral in construct_Y_A is Hermitian"""
om = 2 * np.pi * 3.5e6
tau, delta = get_params(om)
t_initial = 0.0
t_final = 2.0 * tau
q = 5e6

# Get all integrated matrices
integral_V1 = integrate_matrix_A(t_initial, t_final, om, get_V1)
integral_V2 = integrate_matrix_A(t_initial, t_final, om, get_V2)
integral_W0z = integrate_matrix_A(t_initial, t_final, om, get_W0z)
integral_Wz = integrate_matrix_A(t_initial, t_final, om, get_Wz)
print('\n')
print('MAX and MIN values V1: ', np.amax(abs(integral_V1)), np.amin(abs(integral_V1)))
print('MAX and MIN values V2: ', np.amax(abs(integral_V2)), np.amin(abs(integral_V2)))
print('MAX and MIN values W0z: ', np.amax(abs(integral_W0z)), np.amin(abs(integral_W0z)))
print('MAX and MIN values Wz: ', np.amax(abs(integral_Wz)), np.amin(abs(integral_Wz)))

integral_V1 = np.kron(integral_V1, V1_vib)
integral_V2 = np.kron(integral_V2, get_V2_vib(q))
integral_W0z = np.kron(integral_W0z, W0z_vib)
integral_Wz = np.kron(integral_Wz, Wz_vib)

# Combine them
combined_integral = integral_V1 + integral_V2 + integral_W0z + integral_Wz
Y_A = expm(-1j * combined_integral)