import numpy as np
from scipy.optimize import fsolve


def get_delta_renorm(delta, om, delta_rydberg=None):
    if delta_rydberg is None:
        return delta
    rydberg_shift = delta_rydberg - 2 * delta
    return delta + (om ** 2) / (2 * rydberg_shift)  # + (np.sqrt(2) * om) ** 4 / (8 * delta_rydberg ** 3)


def calc_tau(delta, om, delta_rydberg=None):
    delta_renorm = get_delta_renorm(delta, om, delta_rydberg)
    return 2 * np.pi / np.sqrt(2 * om ** 2 + delta_renorm ** 2)


def calc_xi(delta, omega, tau):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    arg = om_0 * tau / 2
    val1 = np.cos(arg) - 1j * delta / om_0 * np.sin(arg)
    val2 = np.cos(arg) + 1j * delta / om_0 * np.sin(arg)
    # (- val1 / val2) * np.exp(1j * delta * tau) # TODO: compare with LP-article
    return np.angle(- val1 / val2)


def to_solve(delta, om, delta_rydberg=None):
    delta = float(delta)  # fsolve below passes delta as ndarray (1,)
    tau = calc_tau(delta, om, delta_rydberg)
    xi = calc_xi(delta, om, tau)
    phi1 = delta * tau + xi + np.pi # - delta * tau
    phi1 = np.angle(np.exp(1j * phi1))  # to ensure that phi is in the (-pi, pi] interval

    delta_renorm = get_delta_renorm(delta, om, delta_rydberg)
    phi2 = delta_renorm * tau

    return phi2 - 2 * phi1 + np.pi


def calc_delta(om, delta_rydberg=None):
    return fsolve(
        to_solve,
        np.array([0.3 * om]),  #  initial guess is close to the delta_R -> infty solution
        args=(om, delta_rydberg),
        factor=0.01 * om,
    )[0]


def calc_params(om, delta_rydberg):
    delta = calc_delta(om, delta_rydberg)
    tau = calc_tau(delta, om, delta_rydberg)
    xi = calc_xi(delta, om, tau)
    return tau, delta, xi
