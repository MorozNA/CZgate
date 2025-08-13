import numpy as np
from tqdm import tqdm
from src.y_operator_deltaR.comparison_full_vs_algo.initial_parameters import calc_rho, rho_TA, rho_TB, rho_full, n
from src.y_operator_deltaR.comparison_full_vs_algo.initial_parameters import YA, YB, U0, full_U
from src.y_operator_deltaR.comparison_full_vs_algo.metrics_fun import generalized_fidelity, negativity
from src.y_operator_deltaR.algorithm.algorithm_fun import one_iteration

iterations = 20
fidelity = []
fidelity_decomp = []
fidelity_spin = []
fidelity_vib = []

# entanglement_measure_full = []
# entanglement_measure_full_decomp = []
# entanglement_measure_alg = []
purity_spin_alg = []
purity_spin_full = []



for i in tqdm(range(iterations)):
    # exact_rho = exact_evolution(exact_rho, U0_perfect)

    rho_full = full_U @ rho_full @ full_U.conj().T
    calc_rho, rho_TA, rho_TB = one_iteration(calc_rho, rho_TA, rho_TB, U0, YA, YB)

    calc_rho = calc_rho / np.trace(calc_rho)
    rho_TA = rho_TA / np.trace(rho_TA)
    rho_TB = rho_TB / np.trace(rho_TB)


    ### CALC FIDELITY FULL
    rho_full_alg = np.kron(calc_rho, np.kron(rho_TA, rho_TB))
    fidelity.append(generalized_fidelity(rho_full, rho_full_alg))

    ### CALC FIDELITY DECOMPOSED
    tensor = rho_full.reshape((9, n, n, 9, n, n))
    rho_full_A = np.einsum('ikmjkm->ij', tensor)
    rho_full_B = np.einsum('ikmilm->kl', tensor)
    rho_full_C = np.einsum('ikmikn->mn', tensor)
    rho_full_decomp = np.kron(rho_full_A, np.kron(rho_full_B, rho_full_C))
    fidelity_decomp.append(generalized_fidelity(rho_full_decomp, rho_full_alg))

    ### CALC FIDELITY SPIN
    fidelity_spin.append(generalized_fidelity(rho_full_A, calc_rho))

    ### CALC FIDELITY VIB
    rho_full_vib = np.einsum('ikmiln->kmln', tensor).reshape((n*n, n*n))
    rho_vib_alg = np.kron(rho_TA, rho_TB)
    fidelity_vib.append(generalized_fidelity(rho_full_vib, rho_vib_alg))

    # CALC PURITY SPIN
    purity_spin_full.append(abs(np.trace(rho_full_A @ rho_full_A)))
    purity_spin_alg.append(abs(np.trace(calc_rho @ calc_rho)))


    print('\n')
    print('Purity full = ', abs(np.trace(rho_full @ rho_full)))
    print('Purity alg = ', abs(np.trace(rho_full_alg @ rho_full_alg)))
    print('\n')
