import numpy as np
from src.y_operator_deltaR.calc_params import get_delta_renorm
from src.y_operator_deltaR.params import get_params
from src.y_operator_deltaR.construct_U0 import construct_U0

# initial_state = np.zeros((9, 1), dtype=complex)
# initial_state[0] = 1.0 / np.sqrt(2)
initial_state_U = np.zeros((9, 1), dtype=complex)
initial_state_V = np.zeros((9, 1), dtype=complex)

initial_state_U[0] = 1.0 / 2
initial_state_U[1] = 1.0 / 2
initial_state_U[3] = 1.0 / 2
initial_state_U[4] = 1.0 / 2
initial_state_V[0] = 1.0 / 2
initial_state_V[1] = 1.0 / 2
initial_state_V[3] = 1.0 / 2
initial_state_V[4] = 1.0 / 2

deltaR_array = 2 * np.pi * np.array([30, 40, 50, 60]) * 1e6
om_array = 2 * np.pi * np.linspace(3, 10, 100) * 1e6

fidelities = [[] for x in range(len(deltaR_array))]
cz_time = [[] for x in range(len(deltaR_array))]

for i in range(len(deltaR_array)):
    delta_R = deltaR_array[i]
    for om in om_array:
        tau, delta, xi = get_params(om, delta_R)
        U0_with_leakage = construct_U0(2 * tau, om, delta_R)

        delta_renorm = get_delta_renorm(delta, om, delta_R)
        phi2 = delta_renorm * tau
        phi1 = (phi2 + np.pi) / 2
        CZ_expected = np.diag([1.0, np.exp(1j * phi1), 1, np.exp(1j * phi1), np.exp(1j * phi2), 1, 1, 1, 1])
        # CZ_expected = construct_U0_elimination(2 * tau, om, delta_R)

        state_with_leakage = U0_with_leakage @ initial_state_U
        state_exact = CZ_expected @ initial_state_V

        fidelity = np.abs(np.vdot(state_exact, state_with_leakage)) ** 2
        fidelities[i].append(fidelity)
        cz_time[i].append(2 * tau)
    if i==0:
        print('delta / omega = ', delta / om)
        print('tau * omega = ', tau * om)
        print('xi = ', xi)
        phase_exact_0 = np.angle(state_exact[0])
        phase_approx_0 = np.angle(state_with_leakage[0])
        phase_error_0 = np.abs(phase_exact_0 - phase_approx_0)
        phase_exact_1 = np.angle(state_exact[1])
        phase_approx_1 = np.angle(state_with_leakage[1])
        phase_error_1 = np.abs(phase_exact_1 - phase_approx_1)
        phase_exact_3 = np.angle(state_exact[3])
        phase_approx_3 = np.angle(state_with_leakage[3])
        phase_error_3 = np.abs(phase_exact_3 - phase_approx_3)
        phase_exact_4 = np.angle(state_exact[4])
        phase_approx_4 = np.angle(state_with_leakage[4])
        phase_error_4 = np.abs(phase_exact_4 - phase_approx_4)
        print('phase error 0: ', phase_error_0)
        print('phase error 1: ', phase_error_1)
        print('phase error 3: ', phase_error_3)
        print('phase error 4: ', phase_error_4)
        # phi1 = (phi2 - np.pi) / 2
        print('phi1_approx = ', phase_approx_1)
        print('phi1_approx = ', phase_approx_3)
        print('(phi2_approx - pi) / 2 = ', (phase_approx_4 - np.pi) / 2)
        print('\n')
        print('phi1_exact = ', phase_exact_1)
        print('phi1_exact = ', phase_exact_3)
        print('(phi2_exact - pi) / 2 = ', (phase_exact_4 - np.pi) / 2)

        import pandas as pd
        # Convert to strings in "a+bj" format
        U_str = np.array([[f"{x.real:.2f}{x.imag:+.2f}j" for x in row] for row in U0_with_leakage])

        # Create DataFrame with state labels
        states = ['|00>', '|01>', '|0r>', '|10>', '|11>', '|1r>', '|r0>', '|r1>', '|rr>']  # Adjust as needed
        df = pd.DataFrame(U_str, index=states, columns=states)

        # Save to Excel
        df.to_excel("U0_with_leakage_complex.xlsx", sheet_name="U0_matrix")
    print(len(fidelities[i]))




import matplotlib.pyplot as plt

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

colors = ['orange', 'green', 'blue', 'gray']
# Upper plot: Fidelities vs om_array/(2π)
for i in range(len(deltaR_array)):
    delta_R_MHz = deltaR_array[i]/(2*np.pi*1e6)  # Convert to MHz for legend
    ax1.plot(om_array/(2*np.pi * 1e6), fidelities[i],
             label=f'$\delta_R/(2π)$ = {delta_R_MHz:.0f} MHz', color=colors[i])

ax1.set_ylabel('Fidelity')
ax1.grid(True)
ax1.set_ylim([0.975, 1.001])
ax1.legend()

# Add ticks to all sides of the first plot
ax1.tick_params(which='both', direction='in', top=True, right=True)
ax1.set_yticks(np.arange(0.975, 1.001, 0.005))  # Y-ticks every 0.005
ax1.minorticks_on()  # Enable minor ticks

# Lower plot: CZ_time vs om_array/(2π)
for i in range(len(deltaR_array)):
    delta_R_MHz = deltaR_array[i]/(2*np.pi*1e6)  # Convert to MHz for legend
    ax2.plot(om_array/(2*np.pi * 1e6), np.array(cz_time[i])*1e6,  # Convert to μs
             label=r'$\delta_R/(2π)$ = {delta_R_MHz:.0f} MHz', color=colors[i])

ax2.set_xlabel(r'$|\Omega|/(2π)$ (MHz)')
ax2.set_ylabel(r'CZ Time ($\mu s$)')
ax2.grid(True)

# Add ticks to all sides of the second plot
ax2.tick_params(which='both', direction='in', top=True, right=True)
ax2.set_xticks([3, 4, 5, 6, 7, 8, 9, 10])
ax2.minorticks_on()  # Enable minor ticks

plt.tight_layout()

# Save the plot in high resolution
plt.savefig(
    'cz_gate_analysis_closer.png',  # File name (supports .png, .pdf, .svg, .jpg)
    dpi=300,                 # High resolution (300 dots per inch)
    bbox_inches='tight',     # Prevent cropping
    transparent=False,       # White background (True for transparent)
    facecolor='white'        # Ensure background is white
)

plt.show()