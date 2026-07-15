import numpy as np
from src.y_operator.config import YOperatorDerived
from src.y_operator.internal import get_V1, get_V2, get_W0z, get_Wz
from src.y_operator.motional import get_V1_mot, get_V2_mot, get_W0z_mot, get_Wz_mot


def get_HA(params: YOperatorDerived):
    HA_int = 0

    V1_int = get_V1(params)
    V1_int = np.kron(V1_int, np.eye(3))
    V1_mot = get_V1_mot(params)
    HA_int += np.kron(V1_int, (np.kron(V1_mot, np.eye(params.n))))

    if params.eff_dict["Q"]:
        V2_int = get_V2(params)
        V2_int = np.kron(V2_int, np.eye(3))
        V2_mot = get_V2_mot(params)
        HA_int += np.kron(V2_int, (np.kron(V2_mot, np.eye(params.n))))

    if params.eff_dict["W"]:
        W0z_int = get_W0z(params, 2.0 * params.tau)
        W0z_int = np.kron(W0z_int, np.eye(3))
        W0z_mot = get_W0z_mot(params)
        HA_int += np.kron(W0z_int, (np.kron(W0z_mot, np.eye(params.n))))

        Wz_int = get_Wz(params, 2.0 * params.tau)
        Wz_int = np.kron(Wz_int, np.eye(3))
        Wz_mot = get_Wz_mot(params)
        HA_int += np.kron(Wz_int, (np.kron(Wz_mot, np.eye(params.n))))

    return HA_int


def get_HB(params: YOperatorDerived):
    HB_int = 0

    V1_int = get_V1(params)
    V1_int = np.kron(np.eye(3), V1_int)
    V1_mot = get_V1_mot(params)
    HB_int += np.kron(V1_int, (np.kron(np.eye(params.n), V1_mot)))

    if params.eff_dict["Q"]:
        V2_int = get_V2(params)
        V2_int = np.kron(np.eye(3), V2_int)
        V2_mot = get_V2_mot(params)
        HB_int += np.kron(V2_int, (np.kron(np.eye(params.n), V2_mot)))

    if params.eff_dict["W"]:
        W0z_int = get_W0z(params, 2.0 * params.tau)
        W0z_int = np.kron(np.eye(3), W0z_int)
        W0z_mot = get_W0z_mot(params)
        HB_int += np.kron(W0z_int, (np.kron(np.eye(params.n), W0z_mot)))

        Wz_int = get_Wz(params, 2.0 * params.tau)
        Wz_int = np.kron(np.eye(3), Wz_int)
        Wz_mot = get_Wz_mot(params)
        HB_int += np.kron(Wz_int, (np.kron(np.eye(params.n), Wz_mot)))

    return HB_int
