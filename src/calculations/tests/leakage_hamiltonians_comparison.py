import numpy as np
from src.y_operator.construct_U0 import get_U
from src.y_operator.construct_U0 import get_U_deltaR
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.config import YOperatorConfig, build_derived
from src.algorithm.other_tools import generalized_fidelity
from dataclasses import replace

om_hz = 5e6
deltaR_inf = None
deltaR_finite = 5e12

cfg_inf = YOperatorConfig(
    om_hz=om_hz,
    delta_rydberg_hz=deltaR_inf,
)
cfg_finite = YOperatorConfig(
    om_hz=om_hz,
    delta_rydberg_hz=deltaR_finite,
)

params_inf = build_derived(cfg_inf)
params_finite = build_derived(cfg_finite)

t_inf = 2 * params_inf.tau
t_finite = 2 * params_finite.tau

print(params_inf.tau / 1e-9)
print(params_finite.tau / 1e-9)
print('\n')
print(params_inf.delta / 1e6)
print(params_finite.delta / 1e6)
print('\n')
print(params_inf.xi / 1e6)
print(params_finite.xi / 1e6)

U_block_no_leakage = get_U(replace(params_inf, om=np.sqrt(2)*params_inf.om), t_inf)
U_block_leakage = get_U_deltaR(params_finite, t_finite)[:2, :2]
difference_block = np.linalg.norm(U_block_no_leakage - U_block_leakage)
print("Norm difference between U_3x3_reduced and U_2x2:", difference_block)


print('\n')
print('==================================================')
print(np.round(U_block_no_leakage, 2))
print('\n')
print(np.round(U_block_leakage, 2))
print('==================================================')


U0_no_leakage = construct_U0(params_inf, t_inf)
U0_leakage = construct_U0(params_finite, t_finite)
difference = np.linalg.norm(U0_no_leakage - U0_leakage)
print("Norm difference between different U_0's:", difference)

print('\n')
print('==================================================')
print(np.round(U0_no_leakage, 2))
print('\n')
print(np.round(U0_leakage, 2))
print('\n')
print(np.round(U0_no_leakage-U0_leakage, 2))
print('==================================================')