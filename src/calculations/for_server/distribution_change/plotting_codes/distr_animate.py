import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src.calculations.plot_temperature.approximate_function import get_temperature
from src.calculations.for_server.get_rho_fun import get_rho_T0
from src.calculations.for_server.distribution_change.temp0_distr_init import n
from src.y_operator_deltaR.params import HBAR, OM_small

rho_TA_list = np.load('temp1_rho_TA_list.npy')
rho_TB_list = np.load('temp1_rho_TA_list.npy')
rho_TA_list = abs(rho_TA_list)
rho_TB_list = abs(rho_TB_list)
temperatures = [get_temperature(np.diag(rho)) for rho in rho_TB_list]
n_iterations = len(temperatures)

# Create the figure and subplots
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.25)

# Left plot: Boltzmann distribution for selected iteration
ax1 = plt.subplot(121)
ax1.set_title('Boltzmann Distribution')
ax1.set_xlabel('Energy Levels')
ax1.set_ylabel('Probability')
# ax1.set_yscale('log')

# Right plot: Temperature evolution
ax2 = plt.subplot(122)
ax2.set_title('Temperature vs Iteration')
ax2.set_xlabel('Iteration Number')
ax2.set_ylabel('Temperature (K)')
ax2.grid(True)

# Plot the temperature evolution line
line, = ax2.plot(range(n_iterations), temperatures, 'b-')
point, = ax2.plot([0], [temperatures[0]], 'ro', markersize=8)  # Highlighted point

# Initial plot (iteration 0)
current_iter = 0
rho = rho_TB_list[current_iter]
probs = rho.real
probs = probs / np.sum(probs)
energies = HBAR * OM_small * (np.arange(len(probs)) + 0.5)

x = np.arange(len(probs))

# Plot probabilities and fit
scatter = ax1.scatter(x, probs, label='Actual', color='blue')
fit_line = ax1.plot(x, np.diag(get_rho_T0(temperatures[current_iter], n)), 'r--', label=f'Fit')#, label=f'Fit (T={temperatures[current_iter]:.2e} K)')
ax1.set_xlim(x[0], x[14])
ax1.set_ylim(0, np.amax(rho_TB_list[0]) + 0.05)
# mask = probs > 1e-10
# if np.sum(mask) >= 2:
#     log_probs = np.log(probs[mask])
#     E_filtered = energies[mask]
#     slope, intercept = np.polyfit(E_filtered, log_probs, 1)
#     fit_line = ax1.plot(energies, np.exp(intercept + slope * energies),
#                         'r-', label=f'Fit (T={temperatures[current_iter]:.2e} K)')
ax1.legend()

# Add slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Iteration', 0, n_iterations - 1, valinit=0, valstep=1)


# Update function for slider
def update(val):
    current_iter = int(slider.val)
    rho = rho_TB_list[current_iter]
    probs = rho.real
    probs = probs / np.sum(probs)

    # Clear and update left plot
    ax1.clear()
    ax1.set_title('Boltzmann Distribution')
    ax1.set_xlabel('Energy Levels')
    ax1.set_ylabel('Probability')
    ax1.set_xlim(x[0], x[14])
    ax1.set_ylim(0, np.amax(rho_TB_list[0]) + 0.05)
    # ax1.set_yscale('log')

    ax1.scatter(x, probs, label='Actual', color='blue')
    ax1.plot(x, np.diag(get_rho_T0(temperatures[current_iter], n)), 'r--', label='Fit') #label=f'Fit (T={temperatures[current_iter]:.2e} K)')
    # mask = probs > 1e-10
    # if np.sum(mask) >= 2:
    #     log_probs = np.log(probs[mask])
    #     E_filtered = energies[mask]
    #     slope, intercept = np.polyfit(E_filtered, log_probs, 1)
    #     ax1.plot(energies, np.exp(intercept + slope * energies),
    #              'r-', label=f'Fit (T={temperatures[current_iter]:.2e} K)')
    ax1.legend()

    # Update right plot highlight
    point.set_data([current_iter], [temperatures[current_iter]])
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()