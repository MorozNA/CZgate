import numpy as np
from scipy.optimize import fsolve


def get_U_perfect(delta, omega, xi, t):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    # TODO: проверить не отличается ли матрица сменой базисных векторов (rr <-> bb, rb <-> br)
    Ubb = (np.cos(om_0 * t / 2) - 1j * delta / om_0 * np.sin(om_0 * t / 2)) * np.exp(1j * delta * t / 2)
    Ubr = 1j * omega / om_0 * np.sin(om_0 * t / 2) * np.exp(-1j * xi + 1j * delta * t / 2)
    Urb = 1j * omega / om_0 * np.sin(om_0 * t / 2) * np.exp(1j * xi - 1j * delta * t / 2)
    Urr = (np.cos(om_0 * t / 2) + 1j * delta / om_0 * np.sin(om_0 * t / 2)) * np.exp(-1j * delta * t / 2)
    U = np.array([[Ubb, Ubr], [Urb, Urr]], dtype=complex)
    return U


def calc_tau(delta, omega, to_print=False):
    tau = 2 * np.pi / np.sqrt(2 * omega ** 2 + delta ** 2)
    if to_print:
        print('tau = ', tau)
    return tau


def calc_xi(delta, omega, tau, to_print=False):
    om_0 = np.sqrt(omega ** 2 + delta ** 2)
    arg = om_0 * tau / 2
    val1 = np.cos(arg) + 1j * delta / om_0 * np.sin(arg)
    val2 = np.cos(arg) - 1j * delta / om_0 * np.sin(arg)
    result_exp = - val1 / val2 * np.exp(-1j * delta * tau)
    if to_print:
        print(r'|e^{i \xi_2}| = ', abs(result_exp))
        print(r'xi_2 = ', np.angle(result_exp))
    return np.angle(result_exp)


def to_solve(delta):
    om = 1
    tau = calc_tau(delta, om)
    xi = calc_xi(delta, om, tau)
    phi1 = np.pi - xi
    phi1 = np.angle(np.exp(1j * phi1))  # так fsolve работает, видимо потому что угол сидит в промежутке (-pi, pi]
    phi2 = -delta * tau
    return (phi2 - np.pi) / 2 - phi1


def calc_delta(to_print=False):
    delta = fsolve(to_solve, np.array([0.3]), factor=0.1)[0]
    if to_print:
        print('Delta = ', delta)
    return delta
