from pydantic import BaseModel, Field
from typing import Optional


class CalculationInput(BaseModel):
    consumo: float = Field(gt=0, description="Consumo base em m³/h")
    k1: float = Field(gt=0, description="Fator do maior dia de consumo")
    k2_factors: list[float] = Field(min_length=24, max_length=24, description="24 fatores K2 horários")
    l_suc: float = Field(ge=0, description="Comprimento de sucção (m)")
    l_rec: float = Field(ge=0, description="Comprimento de recalque (m)")
    sing_suc: float = Field(ge=0, description="Singularidades na sucção")
    sing_rec: float = Field(ge=0, description="Singularidades no recalque")
    h_suc: float = Field(description="Cota da sucção (m)")
    h_rec: float = Field(description="Cota do recalque (m)")
    d_suc: float = Field(gt=0, description="Diâmetro da sucção (mm)")
    d_rec: float = Field(gt=0, description="Diâmetro do recalque (mm)")
    t_fun: float = Field(gt=0, le=24, description="Tempo de funcionamento diário (h)")
    material: str = Field(description="Material da tubulação")
    tarifa: float = Field(ge=0, description="Tarifa kWh em R$")
    selected_pumps: list[str] = Field(default=[], description="Nomes das bombas selecionadas")


class VelocityResult(BaseModel):
    value: float
    status: str  # "ok", "low", "high"


class HeadLossResult(BaseModel):
    linear_suc: float
    linear_rec: float
    singular_suc: float
    singular_rec: float
    total: float


class DiameterResult(BaseModel):
    coef_funcionamento: float
    d_economico: float
    delta_suc: float
    delta_rec: float


class SystemCurvePoint(BaseModel):
    vazao: float
    perda_carga: float


class PumpFitPoint(BaseModel):
    vazao: float
    altura: float


class PumpIntersection(BaseModel):
    pump_name: str
    vazao: float
    altura: float
    above_optimal: bool


class PumpEfficiency(BaseModel):
    pump_name: str
    rendimento: float
    operating_hours: float
    custo_diario: float


class PumpCurveFit(BaseModel):
    pump_name: str
    original_vazao: list[float]
    original_altura: list[float]
    fit_vazao: list[float]
    fit_altura: list[float]
    original_rendimento: list[float]
    fit_rendimento_vazao: list[float]
    fit_rendimento_values: list[float]


class DemandPoint(BaseModel):
    hora: int
    demanda: float


class CalculationResult(BaseModel):
    demand_profile: list[DemandPoint]
    diameters: DiameterResult
    velocity_suc: VelocityResult
    velocity_rec: VelocityResult
    head_losses: HeadLossResult
    dh_geometrico: float
    system_curve: list[SystemCurvePoint]
    q_projeto: float
    h_projeto: float
    pump_fits: list[PumpCurveFit]
    intersections: list[PumpIntersection]
    efficiencies: list[PumpEfficiency]
