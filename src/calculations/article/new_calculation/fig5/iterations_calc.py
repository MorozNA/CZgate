import os
import numpy as np
from tqdm import tqdm
from linear_regression import fit_temperature
from src.algorithm.other_tools import get_rho_T0
from src.algorithm.algorithm_fun_other import one_iteration
from src.algorithm.other_tools import construct_U0_for_trotter
from src.y_operator_deltaR.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator_deltaR.params import lambd_1, lambd_2
from src.y_operator_deltaR.params import get_params


T_muK = 1.8
n = 250
path = f'data/T{T_muK}/n{n}/'
os.makedirs(path, exist_ok=True)

# INITIAL PARAMETERS
temperature = T_muK * 1e-6
om = 2 * np.pi * 5e6
iterations = 100


Q = 2 * np.pi * (1 / lambd_2 - 1 / lambd_1)
delta_R = 2 * np.pi * 50e6
tau, delta, xi = get_params(om, delta_R)
t_initial = 0.0
t_final = 2 * tau
print('Tau = ', tau / 1e-9, ' ns')


# INITIAL STATES
if T_muK==0:
    rho_T0 = np.zeros((n, n), dtype=complex)
    rho_T0[0, 0] = 1.0
else:
    rho_T0 = get_rho_T0(temperature, n)

rho_S0 = np.zeros((9, 9), dtype=complex)
idx = [0, 1, 3, 4]
rho_S0[np.ix_(idx, idx)] = 1/4 * np.ones((4, 4))

rho_elmotA_T0 = np.kron(rho_S0, rho_T0)
rho_elmotB_T0 = np.kron(rho_S0, rho_T0)
rho_el_T0 = np.copy(rho_S0)
rho_T0_A = np.copy(rho_T0)
rho_T0_B = np.copy(rho_T0)

rho_ideal = np.copy(rho_S0)


# EVOLUTION OPERATORS
U01 = construct_U0_for_trotter(tau, om, tau, delta, 0.0, delta_R)
YA1 = construct_Y_A(0.0, tau, om, tau, delta, 0.0, Q, n, delta_R)
YB1 = construct_Y_B(0.0, tau, om, tau, delta, 0.0, Q, n, delta_R)
U02 = construct_U0_for_trotter(tau, om, tau, delta, xi, delta_R)
YA2 = construct_Y_A(tau, 2 * tau, om, tau, delta, xi, Q, n, delta_R)
YB2 = construct_Y_B(tau, 2 * tau, om, tau, delta, xi, Q, n, delta_R)
n_operator = np.diag(np.arange(0, n))


# DATA LISTS
density_matrix_diagonals = np.zeros((iterations, n), dtype=float)
fitted_temperatures = []
n_average = []
fitted_density_matrices = np.zeros((iterations, n), dtype=float)


for i in tqdm(range(iterations)):
    rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U01, YA1, YB1)
    rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B = one_iteration(rho_elmotA_T0, rho_elmotB_T0, rho_el_T0, rho_T0_A, rho_T0_B, U02, YA2, YB2)
    rho_elmotA_T0 /= np.trace(rho_elmotA_T0)
    rho_elmotB_T0 /= np.trace(rho_elmotB_T0)
    rho_el_T0 /= np.trace(rho_el_T0)
    rho_T0_A /= np.trace(rho_T0_A)
    rho_T0_B /= np.trace(rho_T0_B)

    density_matrix_diagonal = np.real(np.diag(rho_T0_A))
    fitted_temperature = fit_temperature(density_matrix_diagonal)
    approximate_density_matrix = get_rho_T0(fitted_temperature, n)

    density_matrix_diagonals[i] = density_matrix_diagonal
    fitted_temperatures.append(fitted_temperature)
    fitted_density_matrices[i] = np.diag(approximate_density_matrix)
    n_average.append(np.real(np.trace(n_operator @ rho_T0_A)))
    print('\n', '<n> = ', np.round(n_average[-1], 3), '\n')


np.savetxt(path + f'dm_diags_T{T_muK}.txt', density_matrix_diagonals, fmt='%.18f')
np.savetxt(path + f'fit_temps_T{T_muK}.txt', fitted_temperatures, fmt='%.18f')
np.savetxt(path + f'dm_fits_T{T_muK}.txt', fitted_density_matrices, fmt='%.18f')
np.savetxt(path + f'n_avg_T{T_muK}.txt', n_average, fmt='%.18f')
