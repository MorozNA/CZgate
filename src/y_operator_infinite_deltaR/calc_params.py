import numpy as np
from scipy.optimize import fsolve


def calc_tau(delta, om):
    tau = 2 * np.pi / np.sqrt(2 * om ** 2 + delta ** 2)
    return tau


def calc_xi(delta, omega, tau):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    arg = om_0 * tau / 2
    val1 = np.cos(arg) - 1j * delta / om_0 * np.sin(arg)
    val2 = np.cos(arg) + 1j * delta / om_0 * np.sin(arg)
    result_exp = - val1 / val2  # * np.exp(1j * delta * tau)
    return np.angle(result_exp)


def to_solve(delta, om):
    tau = calc_tau(delta, om)
    xi = calc_xi(delta, om, tau)
    phi1 = delta * tau + xi + np.pi #- delta * tau
    phi1 = np.angle(np.exp(1j * phi1))  # так fsolve работает, видимо потому что угол сидит в промежутке (-pi, pi]

    phi2 = delta * tau
    # phi2 = 2 * phi + pi
    # phi1 = phi + pi

    return phi2 - 2 * phi1 + np.pi


def calc_delta(om):
    delta = fsolve(to_solve, np.array([0.3 * om]), args=(om,), factor=0.01 * om)[0]
    return delta


def calc_params(om):
    delta = calc_delta(om)
    tau = calc_tau(delta, om)
    xi = calc_xi(delta, om, tau)
    return tau, delta, xi
