import numpy as np
from scipy.optimize import fsolve


def get_delta_renorm(delta, om, delta_rydberg):
    delta_renorm = delta - (np.sqrt(2) * om) ** 2 / (2 * delta_rydberg) #- (np.sqrt(2) * om) ** 4 / (8 * delta_rydberg ** 3)
    return delta_renorm


def calc_tau(delta, om, delta_rydberg=None):
    if delta_rydberg is None:
        delta_renorm = delta
    else:
        delta_renorm = get_delta_renorm(delta, om, delta_rydberg)
    tau = 2 * np.pi / np.sqrt(2 * om ** 2 + delta_renorm ** 2)
    return tau


def calc_xi(delta, omega, tau):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    arg = om_0 * tau / 2
    val1 = np.cos(arg) - 1j * delta / om_0 * np.sin(arg)
    val2 = np.cos(arg) + 1j * delta / om_0 * np.sin(arg)
    result_exp = - val1 / val2 * np.exp(1j * delta * tau)
    return -np.angle(result_exp)


def to_solve(delta, om, delta_rydberg=None):
    tau = calc_tau(delta, om, delta_rydberg)
    xi = calc_xi(delta, om, tau)
    phi1 = xi + np.pi
    phi1 = np.angle(np.exp(1j * phi1))  # так fsolve работает, видимо потому что угол сидит в промежутке (-pi, pi]

    if delta_rydberg is None:
        delta_renorm = delta
    else:
        delta_renorm = get_delta_renorm(delta, om, delta_rydberg)
    phi2 = - delta_renorm * tau

    return (phi2 - np.pi) / 2 - phi1


def calc_delta(om, delta_rydberg=None):
    delta = fsolve(to_solve, np.array([0.3 * om]), args=(om, delta_rydberg,), factor=0.1 * om)[0]
    return delta


def calc_params(om, delta_rydberg=None):
    delta = calc_delta(om, delta_rydberg)
    tau = calc_tau(delta, om, delta_rydberg)
    xi = calc_xi(delta, om, tau)
    return tau, delta, xi
