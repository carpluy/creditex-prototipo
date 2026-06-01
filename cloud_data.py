import pandas as pd
import time
import random
from datetime import datetime
from pathlib import Path

CLOUD_CONFIG = {
    "provider":  "Azure SQL Database",
    "host":      "creditex-prod.database.windows.net",
    "database":  "telares_produccion",
    "schema":    "dbo",
    "table":     "registros_operativos",
    "user":      "uni_readonly",
    "region":    "South America (Lima)",
    "ssl":       True,
}

_BASE = Path(__file__).parent
_EXCEL_NAMES = [
    "Dataset_Tejido_Plano_CREDITEX_IA.xlsx",
    "dataset_creditex.xlsx",
    "creditex.xlsx",
]

def _load_excel():
    for name in _EXCEL_NAMES:
        p = _BASE / name
        if p.exists():
            return pd.read_excel(p)
    raise FileNotFoundError("No se encontró el archivo Excel de CREDITEX.")

_DF_CACHE = None

def _get_df():
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = _load_excel()
    return _DF_CACHE

def simular_latencia(min_ms=80, max_ms=350):
    time.sleep(random.uniform(min_ms, max_ms) / 1000)

def get_connection_status():
    simular_latencia(150, 400)
    df = _get_df()
    return {
        "connected":       True,
        "host":            CLOUD_CONFIG["host"],
        "database":        CLOUD_CONFIG["database"],
        "latency_ms":      random.randint(90, 280),
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_available":  len(df),
        "telares_activos": df["Telar"].nunique(),
        "rango_fechas":    f"{df['Fecha'].min()} → {df['Fecha'].max()}",
        "ssl":             "TLS 1.2",
        "status":          "OK",
    }

def fetch_records(n=2000, turno=None, tipo_tela=None, telar=None,
                  nivel_riesgo=None, cliente=None):
    simular_latencia(200, 500)
    df = _get_df().copy()
    if turno        and turno        != "Todos": df = df[df["Turno"] == turno]
    if telar        and telar        != "Todos": df = df[df["Telar"] == telar]
    if nivel_riesgo and nivel_riesgo != "Todos": df = df[df["Nivel_Riesgo"] == nivel_riesgo]
    if cliente      and cliente      != "Todos": df = df[df["Cliente"] == cliente]
    return df.head(n).reset_index(drop=True)

def fetch_latest_reading(telar="T01"):
    simular_latencia(50, 150)
    df = _get_df()
    rows = df[df["Telar"] == telar]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "telar":        row["Telar"],
        "fecha":        row["Fecha"],
        "timestamp":    datetime.now().strftime("%H:%M:%S"),
        "temperatura":  row["Temperatura"],
        "humedad":      row["Humedad"],
        "rpm_telar":    row["RPM_Telar"],
        "eficiencia":   row["Eficiencia_Telar"],
        "nivel_riesgo": row["Nivel_Riesgo"],
        "tipo_defecto": row["Tipo_Defecto"],
        "fuente":       f"{CLOUD_CONFIG['host']} / {CLOUD_CONFIG['database']}",
    }

def get_summary_stats():
    simular_latencia(150, 350)
    df = _get_df()
    total = len(df)
    pct_alto      = round(100 * (df["Nivel_Riesgo"] == "Alto").sum() / total, 1)
    pct_rechazado = round(100 * (df["Resultado_Inspeccion"] == "Rechazado").sum() / total, 1)
    turno_riesgo  = df[df["Nivel_Riesgo"] == "Alto"]["Turno"].mode()[0]
    telar_incid   = df[df["Resultado_Inspeccion"] == "Rechazado"]["Telar"].mode()[0]
    cliente_rec   = df[df["Reclamo_Cliente"] == 1]["Cliente"].mode()[0] if "Reclamo_Cliente" in df.columns else "N/A"
    return {
        "total_registros":      total,
        "pct_nivel_alto":       pct_alto,
        "pct_rechazado":        pct_rechazado,
        "turno_mayor_riesgo":   turno_riesgo,
        "telar_mayor_incid":    telar_incid,
        "cliente_mas_reclamos": cliente_rec,
        "periodo":              "Dataset completo CREDITEX",
    }

def get_filter_options():
    simular_latencia(50, 120)
    df = _get_df()
    return {
        "telares":  ["Todos"] + sorted(df["Telar"].dropna().unique().tolist()),
        "turnos":   ["Todos"] + df["Turno"].dropna().unique().tolist(),
        "clientes": ["Todos"] + sorted(df["Cliente"].dropna().unique().tolist()),
        "niveles":  ["Todos"] + df["Nivel_Riesgo"].dropna().unique().tolist(),
    }
