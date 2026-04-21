import numpy as np
from scipy.optimize import fsolve


def get_U(delta, omega, xi, t):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    Ubb = (np.cos(om_0 * t / 2) + 1j * delta / om_0 * np.sin(om_0 * t / 2)) * np.exp(-1j * delta * t / 2)
    Ubr = 1j * omega / om_0 * np.sin(om_0 * t / 2) * np.exp(1j * xi - 1j * delta * t / 2)
    Urb = 1j * omega / om_0 * np.sin(om_0 * t / 2) * np.exp(-1j * xi + 1j * delta * t / 2)
    Urr = (np.cos(om_0 * t / 2) - 1j * delta / om_0 * np.sin(om_0 * t / 2)) * np.exp(1j * delta * t / 2)
    U = np.array([[Ubb, Ubr], [Urb, Urr]], dtype=complex)
    return U


def calc_tau(delta, om, delta_R=None):
    if delta_R is None:
        delta_renorm = delta
    else:
        delta_renorm = delta - om ** 2 / 2 / delta_R
    tau = 2 * np.pi / np.sqrt(2 * om ** 2 + delta_renorm ** 2)
    return tau


def calc_xi(delta, omega, tau):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    arg = om_0 * tau / 2
    val1 = np.cos(arg) - 1j * delta / om_0 * np.sin(arg)
    val2 = np.cos(arg) + 1j * delta / om_0 * np.sin(arg)
    result_exp = - val1 / val2 * np.exp(1j * delta * tau)
    return -np.angle(result_exp)


def to_solve(delta, om, delta_R=None):
    tau = calc_tau(delta, om, delta_R)
    xi = calc_xi(delta, om, tau)
    phi1 = xi + np.pi
    phi1 = np.angle(np.exp(1j * phi1))  # fix phi in the interval (-pi, pi] for 'fsolve' to work
    if delta_R is None:
        delta_renorm = delta
    else:
        delta_renorm = delta - om ** 2 / 2 / delta_R
    phi2 = -delta_renorm * tau
    return (phi2 - np.pi) / 2 - phi1


def calc_delta(om, delta_R=None):
    delta = fsolve(to_solve, np.array([0.3 * om]), args=(om, delta_R,), factor=0.1 * om)[0]
    return delta
