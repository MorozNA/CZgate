import numpy as np
from dataclasses import dataclass
from src.y_operator.constants import HBAR, M
from src.y_operator.calc_params import calc_params

@dataclass
class YOperatorConfig:
    # values user can tweak (in Hz / meters etc.)
    DELTA_b_hz: float = -2.5e6
    DELTA_r_hz: float = -2.5e6
    DELTA_a_hz: float = -0.7e6

    W_INT_CONSTANT: float = 1.0
    w01: float = 10e-6
    w02: float = 10e-6

    Q_INT_CONSTANT: float = 1.0
    lambd_1: float = 795e-9
    lambd_2: float = 480e-9
    OM_small_hz: float = 7.158e3

    om_hz: float = 5e6
    delta_rydberg_hz: float | None = 50e6
    n: int = 30


@dataclass
class YOperatorDerived:
    Q_INT_CONSTANT: float
    W_INT_CONSTANT: float
    OM_small: float
    DELTA_a: float
    DELTA_b: float
    DELTA_r: float
    p0z: float
    Z_ast: float
    z_ij_matrix: np.ndarray
    x_ij_matrix: np.ndarray
    q: float
    delta_rydberg: float | None
    om: float
    tau: float
    delta: float
    xi: float
    n: int

def build_derived(cfg: YOperatorConfig) -> YOperatorDerived:
    OM_small = 2 * np.pi * cfg.OM_small_hz
    DELTA_b = 2 * np.pi * cfg.DELTA_b_hz
    DELTA_r = 2 * np.pi * cfg.DELTA_r_hz
    DELTA_a = 2 * np.pi * cfg.DELTA_a_hz

    z_r1 = np.pi * cfg.w01 ** 2 / cfg.lambd_1
    z_r2 = np.pi * cfg.w02 ** 2 / cfg.lambd_2
    p0z = np.sqrt(HBAR * M * OM_small / 2.0)  # kg * m / s; coeff '2' here changes momentum matrices
    Z_ast = 2 * z_r1 * z_r2 / (z_r1 + z_r2)

    z_ij = np.zeros((3, 3), dtype=complex)
    z_ij[0, 0] = 1 / z_r1 ** 2
    z_ij[1, 1] = 1 / z_r1 ** 2
    z_ij[1, 2] = 0.5 / (z_r1 ** 2) + 0.5 / (z_r2 ** 2)
    z_ij[2, 1] = z_ij[1, 2]
    z_ij[2, 2] = 1 / z_r2 ** 2

    x_ij = np.zeros((3, 3), dtype=complex)
    x_ij[0, 0] = 1 / cfg.w01 ** 2
    x_ij[1, 1] = 1 / cfg.w01 ** 2
    x_ij[1, 2] = 1 / (cfg.w01 ** 2) + 1 / (cfg.w02 ** 2)
    x_ij[2, 1] = x_ij[1, 2]
    x_ij[2, 2] = 1 / cfg.w01 ** 2

    q = 2 * np.pi * (1 / cfg.lambd_2 - 1 / cfg.lambd_1)

    if cfg.delta_rydberg_hz is not None:
        delta_rydberg = 2 * np.pi * cfg.delta_rydberg_hz
    else:
        delta_rydberg = None

    om = 2 * np.pi * cfg.om_hz
    tau, delta, xi = calc_params(om, delta_rydberg)


    return YOperatorDerived(
        Q_INT_CONSTANT=cfg.Q_INT_CONSTANT,
        W_INT_CONSTANT=cfg.W_INT_CONSTANT,
        OM_small=OM_small,
        DELTA_a=DELTA_a,
        DELTA_b=DELTA_b,
        DELTA_r=DELTA_r,
        p0z=p0z,
        Z_ast=Z_ast,
        z_ij_matrix=z_ij,
        x_ij_matrix=x_ij,
        q=q,
        delta_rydberg=delta_rydberg,
        om=om,
        tau=tau,
        delta=delta,
        xi=xi,
        n=cfg.n
    )