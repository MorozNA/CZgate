import numpy as np
import matplotlib.pyplot as plt
from calc_file import fidelities, om_array, tau_times_omega_arr, delta_rydberg

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

colors = ['orange', 'green', 'blue', 'gray']
# Upper plot
delta_R_MHz = delta_rydberg/(2*np.pi*1e6)
ax1.plot(om_array/(2*np.pi * 1e6), fidelities, label=f'$\delta_R/(2π)$ = {delta_R_MHz:.0f} MHz', color='orange')

ax1.set_ylabel('Fidelity')
ax1.grid(True)
ax1.set_ylim([0.975, 1.001])
ax1.legend()

# Add ticks to all sides of the first plot
ax1.tick_params(which='both', direction='in', top=True, right=True)
ax1.set_yticks(np.arange(0.975, 1.001, 0.005))  # Y-ticks every 0.005
ax1.minorticks_on()  # Enable minor ticks

# Lower plot
tau = tau_times_omega_arr / om_array
ax2.plot(om_array/(2*np.pi * 1e6), np.array(2 * tau)*1e6, label=f'$\delta_R/(2π)$ = {delta_R_MHz:.0f} MHz', color='orange')

ax2.set_xlabel(r'$|\Omega|/(2π)$ (MHz)')
ax2.set_ylabel('CZ Time ($\mu s$)')
ax2.grid(True)

# Add ticks to all sides of the second plot
ax2.tick_params(which='both', direction='in', top=True, right=True)
ax2.set_xticks([3, 4, 5, 6, 7, 8, 9, 10])
ax2.minorticks_on()  # Enable minor ticks

plt.tight_layout()

plt.show()