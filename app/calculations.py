import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from .config import HAZEN_WILLIAMS_COEFFICIENTS, GRAVITY, WATER_DENSITY, PI, VELOCITY_MIN, VELOCITY_MAX


def calc_demand_profile(k2_factors: list[float], consumo: float, k1: float) -> list[dict]:
    return [{"hora": i, "demanda": k2_factors[i] * k1 * consumo} for i in range(24)]


def calc_economic_diameter(t_fun: float, consumo: float, k1: float) -> dict:
    coef_fun = t_fun / 24.0
    d_econ = 1.3 * (coef_fun ** 0.25) * ((consumo * k1 / 3600.0) ** 0.5) * 1000.0
    return {"coef_funcionamento": round(coef_fun, 4), "d_economico": round(d_econ, 2)}


def calc_velocity(flow_m3h: float, diameter_mm: float) -> dict:
    radius_m = diameter_mm / 2000.0
    area = PI * radius_m ** 2
    velocity = flow_m3h / (3600.0 * area)
    if velocity > VELOCITY_MAX:
        status = "high"
    elif velocity < VELOCITY_MIN:
        status = "low"
    else:
        status = "ok"
    return {"value": round(velocity, 4), "status": status}


def calc_hazen_williams_loss(flow_m3h: float, diameter_mm: float, material: str, length_m: float) -> float:
    c = HAZEN_WILLIAMS_COEFFICIENTS[material]
    d_m = diameter_mm / 1000.0
    q_m3s = flow_m3h / 3600.0
    if d_m <= 0 or c <= 0:
        return 0.0
    j = 10.643 * (q_m3s ** 1.85) / ((c ** 1.85) * (d_m ** 4.87))
    return j * length_m


def calc_singular_loss(velocity: float, n_singularities: float) -> float:
    return (velocity ** 2 * n_singularities) / (2 * GRAVITY)


def calc_system_curve(dh_geo: float, coef_k: float, q_max: float, n_points: int = 500) -> list[dict]:
    vazao = np.linspace(0, q_max, n_points)
    perda = dh_geo + coef_k * vazao ** 1.852
    return [{"vazao": round(float(v), 3), "perda_carga": round(float(p), 4)} for v, p in zip(vazao, perda)]


def fit_pump_curve(vazao: np.ndarray, altura: np.ndarray) -> tuple:
    mask = ~np.isnan(vazao) & ~np.isnan(altura)
    X = vazao[mask].reshape(-1, 1)
    y = altura[mask].reshape(-1, 1)
    if len(X) < 3:
        return None, None, None
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    def predict(x_vals):
        x_arr = np.array(x_vals).reshape(-1, 1)
        return model.predict(poly.transform(x_arr)).flatten()

    x_fit = np.linspace(float(X.min()), float(X.max()), 100)
    y_fit = predict(x_fit)
    return predict, x_fit.tolist(), y_fit.tolist()


def find_intersection(system_vazao: np.ndarray, system_h: np.ndarray, pump_predict, q_max: float) -> dict | None:
    y_bomba = pump_predict(system_vazao)
    diff = y_bomba - system_h
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None
    idx = sign_changes[0]
    x0, x1 = system_vazao[idx], system_vazao[idx + 1]
    y0, y1 = diff[idx], diff[idx + 1]
    if (y1 - y0) == 0:
        return None
    x_int = x0 - y0 * (x1 - x0) / (y1 - y0)
    y_int = float(np.interp(x_int, system_vazao, system_h))
    return {"vazao": round(float(x_int), 2), "altura": round(y_int, 2)}


def fit_efficiency_curve(vazao: np.ndarray, rendimento: np.ndarray):
    mask = ~np.isnan(vazao) & ~np.isnan(rendimento)
    X = vazao[mask].reshape(-1, 1)
    y = rendimento[mask]
    if len(X) < 3:
        return None, [], []
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    def predict(x_vals):
        x_arr = np.array(x_vals).reshape(-1, 1)
        return model.predict(poly.transform(x_arr)).flatten()

    x_fit = np.linspace(float(X.min()), float(X.max()), 100)
    y_fit = predict(x_fit)
    return predict, x_fit.tolist(), y_fit.tolist()


def calc_energy_cost(
    q_bomba: float,
    h_bomba: float,
    rendimento_pct: float,
    consumo: float,
    k1: float,
    tarifa: float,
) -> dict:
    """
    P_hidraulica = rho * g * Q * H  (W)
    P_eixo = P_hidraulica / (eta/100)
    Horas = (Consumo * k1 * 24) / Q_bomba
    Custo = P_eixo * Horas * Tarifa / 1000  (R$/dia)
    """
    if rendimento_pct <= 0 or q_bomba <= 0:
        return {"operating_hours": 0.0, "custo_diario": 0.0}
    q_m3s = q_bomba / 3600.0
    p_hydraulic = WATER_DENSITY * GRAVITY * q_m3s * h_bomba
    p_shaft = p_hydraulic / (rendimento_pct / 100.0)
    operating_hours = (consumo * k1 * 24.0) / q_bomba
    custo_diario = (p_shaft / 1000.0) * operating_hours * tarifa
    return {
        "operating_hours": round(operating_hours, 2),
        "custo_diario": round(custo_diario, 2),
    }
