import math

HAZEN_WILLIAMS_COEFFICIENTS = {
    "Aço": 135,
    "Aço Galvanizado": 125,
    "Cobre": 130,
    "Chumbo": 130,
    "Latão": 130,
    "PVC": 140,
    "Ferro Fundido Revestido": 130,
    "Ferro Fundido Novo": 125,
    "Ferro Fundido Usado": 90,
    "Concreto": 120,
}

DEFAULT_K2_FACTORS = [
    0.74, 0.66, 0.61, 0.59, 0.58, 0.60, 0.77, 0.94, 1.08, 1.20,
    1.31, 1.35, 1.32, 1.26, 1.20, 1.14, 1.12, 1.13, 1.21, 1.26,
    1.17, 1.07, 0.96, 0.86,
]

MATERIALS = list(HAZEN_WILLIAMS_COEFFICIENTS.keys())

GRAVITY = 9.81
WATER_DENSITY = 1000  # kg/m³
PI = math.pi

VELOCITY_MIN = 0.6  # m/s
VELOCITY_MAX = 3.0  # m/s
