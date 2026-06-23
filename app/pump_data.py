import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "Bomba.xlsx"


def get_pump_catalog() -> list[dict]:
    df = pd.read_excel(DATA_PATH, sheet_name=0)
    pumps = []
    for i in range(len(df)):
        name = f"{df.iloc[i, 0]} - {df.iloc[i, 1]}"
        sheet_name = name.replace("/", "_").replace("cv", "")
        pumps.append({"name": name, "sheet_name": sheet_name})
    return pumps


def get_pump_curve(sheet_name: str) -> dict | None:
    try:
        df = pd.read_excel(DATA_PATH, sheet_name=sheet_name, index_col=None, header=None)
    except Exception:
        return None

    try:
        vazao_raw = df.iloc[0, 1:].values.astype(float)
        altura_raw = df.iloc[1, 1:].values.astype(float)
        rendimento_raw = df.iloc[2, 1:].values.astype(float) if len(df) > 2 else np.array([])
    except (IndexError, ValueError):
        return None

    vazao = vazao_raw[~np.isnan(vazao_raw)]
    altura = altura_raw[~np.isnan(altura_raw)]

    min_len = min(len(vazao), len(altura))
    vazao = vazao[:min_len]
    altura = altura[:min_len]

    rend = rendimento_raw[~np.isnan(rendimento_raw)] if len(rendimento_raw) > 0 else np.array([])
    rend_len = min(len(rend), min_len)
    rend = rend[:rend_len]

    return {
        "vazao": vazao.tolist(),
        "altura": altura.tolist(),
        "rendimento": rend.tolist(),
    }
