import os
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from sqlalchemy.orm import Session

from .config import HAZEN_WILLIAMS_COEFFICIENTS, DEFAULT_K2_FACTORS, MATERIALS
from .models import CalculationInput, CalculationResult, CustomPumpInput
from typing import Optional
from .calculations import (
    calc_demand_profile,
    calc_economic_diameter,
    calc_velocity,
    calc_hazen_williams_loss,
    calc_singular_loss,
    calc_system_curve,
    fit_pump_curve,
    find_intersection,
    fit_efficiency_curve,
    calc_energy_cost,
)
from .pump_data import get_pump_catalog, get_pump_curve
from .database import init_db, get_db, CustomPump

app = FastAPI(title="Otimização de Bombas", version="2.0.0")


@app.on_event("startup")
def startup():
    init_db()

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/config")
async def get_config():
    return {
        "materials": MATERIALS,
        "k2_defaults": DEFAULT_K2_FACTORS,
        "hazen_williams": HAZEN_WILLIAMS_COEFFICIENTS,
    }


@app.get("/api/pumps")
async def list_pumps(db: Session = Depends(get_db)):
    catalog = get_pump_catalog()
    if db is not None:
        custom = db.query(CustomPump).order_by(CustomPump.name).all()
        for p in custom:
            catalog.append({"name": p.name, "sheet_name": f"custom:{p.id}", "custom": True})
    return catalog


@app.get("/api/pumps/{sheet_name}/curve")
async def pump_curve(sheet_name: str, db: Session = Depends(get_db)):
    if sheet_name.startswith("custom:") and db is not None:
        pump_id = int(sheet_name.split(":")[1])
        p = db.query(CustomPump).filter(CustomPump.id == pump_id).first()
        if p:
            return {"vazao": p.vazao, "altura": p.altura, "rendimento": p.rendimento or []}
        raise HTTPException(status_code=404, detail="Bomba customizada não encontrada")
    data = get_pump_curve(sheet_name)
    if data is None:
        raise HTTPException(status_code=404, detail="Bomba não encontrada")
    return data


@app.get("/api/custom-pumps")
async def list_custom_pumps(db: Session = Depends(get_db)):
    if db is None:
        return []
    pumps = db.query(CustomPump).order_by(CustomPump.name).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "vazao": p.vazao,
            "altura": p.altura,
            "rendimento": p.rendimento or [],
        }
        for p in pumps
    ]


@app.post("/api/custom-pumps", status_code=201)
async def create_custom_pump(pump: CustomPumpInput, db: Session = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado")
    if len(pump.vazao) != len(pump.altura):
        raise HTTPException(status_code=400, detail="Vazão e altura devem ter o mesmo número de pontos")
    if pump.rendimento and len(pump.rendimento) != len(pump.vazao):
        raise HTTPException(status_code=400, detail="Rendimento deve ter o mesmo número de pontos que vazão")
    existing = db.query(CustomPump).filter(CustomPump.name == pump.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Bomba '{pump.name}' já existe")
    row = CustomPump(
        name=pump.name,
        vazao=pump.vazao,
        altura=pump.altura,
        rendimento=pump.rendimento or [],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "name": row.name}


@app.delete("/api/custom-pumps/{pump_id}")
async def delete_custom_pump(pump_id: int, db: Session = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Banco de dados não configurado")
    row = db.query(CustomPump).filter(CustomPump.id == pump_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Bomba não encontrada")
    db.delete(row)
    db.commit()
    return {"ok": True}


@app.post("/api/calculate", response_model=CalculationResult)
async def calculate(inp: CalculationInput):
    if inp.material not in HAZEN_WILLIAMS_COEFFICIENTS:
        raise HTTPException(status_code=400, detail=f"Material '{inp.material}' não encontrado")

    q_design = inp.consumo * inp.k1

    demand = calc_demand_profile(inp.k2_factors, inp.consumo, inp.k1)

    diam = calc_economic_diameter(inp.t_fun, inp.consumo, inp.k1)
    d_econ = diam["d_economico"]
    delta_suc = round(inp.d_suc / d_econ - 1, 4) if d_econ > 0 else 0
    delta_rec = round(inp.d_rec / d_econ - 1, 4) if d_econ > 0 else 0

    vel_suc = calc_velocity(q_design, inp.d_suc)
    vel_rec = calc_velocity(q_design, inp.d_rec)

    # BUG FIX: suction uses d_suc (was d_rec in original)
    loss_lin_suc = calc_hazen_williams_loss(q_design, inp.d_suc, inp.material, inp.l_suc)
    loss_lin_rec = calc_hazen_williams_loss(q_design, inp.d_rec, inp.material, inp.l_rec)
    loss_sing_suc = calc_singular_loss(vel_suc["value"], inp.sing_suc)
    loss_sing_rec = calc_singular_loss(vel_rec["value"], inp.sing_rec)
    loss_total = loss_lin_suc + loss_lin_rec + loss_sing_suc + loss_sing_rec

    dh_geo = inp.h_rec - inp.h_suc

    q_projeto = q_design * 24.0 / inp.t_fun
    coef_k = loss_total / (q_design ** 1.852) if q_design > 0 else 0
    h_projeto = dh_geo + coef_k * (q_projeto ** 1.852)

    q_max = q_projeto * 4
    system_curve = calc_system_curve(dh_geo, coef_k, q_max, 500)
    sys_vazao = np.array([p["vazao"] for p in system_curve])
    sys_h = np.array([p["perda_carga"] for p in system_curve])

    pump_fits = []
    intersections = []
    efficiencies = []

    for pump_sheet in inp.selected_pumps:
        curve_data = get_pump_curve(pump_sheet)
        if curve_data is None:
            continue

        vazao = np.array(curve_data["vazao"])
        altura = np.array(curve_data["altura"])
        rendimento = np.array(curve_data["rendimento"])

        predict_h, fit_x, fit_y = fit_pump_curve(vazao, altura)
        if predict_h is None:
            continue

        eff_predict, eff_fit_x, eff_fit_y = fit_efficiency_curve(vazao, rendimento) if len(rendimento) >= 3 else (None, [], [])

        pump_fits.append({
            "pump_name": pump_sheet,
            "original_vazao": vazao.tolist(),
            "original_altura": altura.tolist(),
            "fit_vazao": fit_x,
            "fit_altura": fit_y,
            "original_rendimento": rendimento.tolist(),
            "fit_rendimento_vazao": eff_fit_x,
            "fit_rendimento_values": eff_fit_y,
        })

        intersection = find_intersection(sys_vazao, sys_h, predict_h, q_max)
        if intersection:
            above = intersection["vazao"] > q_projeto
            intersections.append({
                "pump_name": pump_sheet,
                "vazao": intersection["vazao"],
                "altura": intersection["altura"],
                "above_optimal": above,
            })

            if eff_predict is not None:
                rend_at_point = float(eff_predict([intersection["vazao"]])[0])
                operating_hours = (inp.consumo * inp.k1 * 24) / intersection["vazao"]
                if operating_hours <= 24:
                    cost = calc_energy_cost(
                        q_bomba=intersection["vazao"],
                        h_bomba=intersection["altura"],
                        rendimento_pct=rend_at_point,
                        consumo=inp.consumo,
                        k1=inp.k1,
                        tarifa=inp.tarifa,
                    )
                    efficiencies.append({
                        "pump_name": pump_sheet,
                        "rendimento": round(rend_at_point, 2),
                        "operating_hours": cost["operating_hours"],
                        "custo_diario": cost["custo_diario"],
                    })

    return CalculationResult(
        demand_profile=demand,
        diameters={
            "coef_funcionamento": diam["coef_funcionamento"],
            "d_economico": diam["d_economico"],
            "delta_suc": delta_suc,
            "delta_rec": delta_rec,
        },
        velocity_suc=vel_suc,
        velocity_rec=vel_rec,
        head_losses={
            "linear_suc": round(loss_lin_suc, 4),
            "linear_rec": round(loss_lin_rec, 4),
            "singular_suc": round(loss_sing_suc, 4),
            "singular_rec": round(loss_sing_rec, 4),
            "total": round(loss_total, 4),
        },
        dh_geometrico=round(dh_geo, 2),
        system_curve=system_curve,
        q_projeto=round(q_projeto, 2),
        h_projeto=round(h_projeto, 2),
        pump_fits=pump_fits,
        intersections=intersections,
        efficiencies=efficiencies,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
