"""
cloud_data.py — Conexión simulada a base de datos en la nube (CREDITEX)

Usa una base de datos SQLite local (creditex_cloud.db) cargada con los
datos reales del Excel de CREDITEX. Simula exactamente lo que sería
una conexión a Azure SQL / AWS RDS / Google Cloud SQL.

En producción real, reemplazarías sqlite3 con pyodbc o sqlalchemy
y cambiarías la cadena de conexión a las credenciales de CREDITEX.
"""

import sqlite3
import pandas as pd
import time
import random
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE CONEXIÓN
# En producción sería:
#   SERVER   = "creditex-prod.database.windows.net"
#   DATABASE = "telares_produccion"
#   USER     = "uni_readonly"
#   PASSWORD = "***"
#   conn_str = f"DRIVER={{ODBC Driver 18}};SERVER={SERVER};DATABASE={DATABASE};UID={USER};PWD={PASSWORD}"
# ─────────────────────────────────────────────
CLOUD_CONFIG = {
    "provider":  "Azure SQL Database (simulado con datos reales CREDITEX)",
    "host":      "creditex-prod.database.windows.net",
    "database":  "telares_produccion",
    "schema":    "dbo",
    "table":     "registros_operativos",
    "user":      "uni_readonly",
    "region":    "South America (Lima)",
    "ssl":       True,
}

DB_PATH = Path(__file__).parent / "creditex_cloud.db"


def _get_conn():
    """Retorna conexión a la BD local (simula conexión cloud)."""
    return sqlite3.connect(DB_PATH)


def simular_latencia(min_ms=80, max_ms=350):
    time.sleep(random.uniform(min_ms, max_ms) / 1000)


# ─────────────────────────────────────────────
# FUNCIONES PÚBLICAS
# ─────────────────────────────────────────────

def get_connection_status():
    """
    Verifica conexión y retorna metadata de la BD.
    Simula: SELECT @@VERSION, COUNT(*) FROM registros_operativos
    """
    simular_latencia(150, 400)
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM registros_operativos").fetchone()[0]
    telares = conn.execute("SELECT COUNT(DISTINCT Telar) FROM registros_operativos").fetchone()[0]
    fecha_min = conn.execute("SELECT MIN(Fecha) FROM registros_operativos").fetchone()[0]
    fecha_max = conn.execute("SELECT MAX(Fecha) FROM registros_operativos").fetchone()[0]
    conn.close()
    return {
        "connected":       True,
        "host":            CLOUD_CONFIG["host"],
        "database":        CLOUD_CONFIG["database"],
        "latency_ms":      random.randint(90, 280),
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_available":  total,
        "telares_activos": telares,
        "rango_fechas":    f"{fecha_min} → {fecha_max}",
        "ssl":             "TLS 1.2",
        "status":          "OK",
    }


def fetch_records(n=2000, turno=None, tipo_tela=None, telar=None,
                  nivel_riesgo=None, cliente=None):
    """
    Ejecuta un SELECT con filtros opcionales.

    Equivalente SQL en producción:
        SELECT TOP {n} * FROM dbo.registros_operativos
        WHERE turno = ? AND ...
        ORDER BY Fecha DESC
    """
    simular_latencia(200, 600)

    where, params = [], []
    if turno        and turno        != "Todos": where.append("Turno = ?");       params.append(turno)
    if tipo_tela    and tipo_tela    != "Todos": where.append("Descripcion_Articulo LIKE ?"); params.append(f"%{tipo_tela}%")
    if telar        and telar        != "Todos": where.append("Telar = ?");       params.append(telar)
    if nivel_riesgo and nivel_riesgo != "Todos": where.append("Nivel_Riesgo = ?"); params.append(nivel_riesgo)
    if cliente      and cliente      != "Todos": where.append("Cliente = ?");     params.append(cliente)

    sql = "SELECT * FROM registros_operativos"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY Fecha DESC LIMIT {n}"

    conn = _get_conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def fetch_latest_reading(telar="T01"):
    """
    Último registro del telar indicado.
    Equivalente: SELECT TOP 1 * FROM ... WHERE Telar=? ORDER BY Fecha DESC
    """
    simular_latencia(50, 150)
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM registros_operativos WHERE Telar = ? ORDER BY Fecha DESC LIMIT 1",
        conn, params=[telar]
    )
    conn.close()
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "telar":       row["Telar"],
        "fecha":       row["Fecha"],
        "timestamp":   datetime.now().strftime("%H:%M:%S"),
        "temperatura": row["Temperatura"],
        "humedad":     row["Humedad"],
        "rpm_telar":   row["RPM_Telar"],
        "eficiencia":  row["Eficiencia_Telar"],
        "nivel_riesgo":row["Nivel_Riesgo"],
        "tipo_defecto":row["Tipo_Defecto"],
        "fuente":      f"{CLOUD_CONFIG['host']} / {CLOUD_CONFIG['database']}",
    }


def get_summary_stats():
    """
    KPIs agregados desde la BD real.
    Equivalente: múltiples GROUP BY y COUNT sobre la tabla.
    """
    simular_latencia(150, 350)
    conn = _get_conn()

    total = conn.execute("SELECT COUNT(*) FROM registros_operativos").fetchone()[0]
    pct_alto = conn.execute(
        "SELECT ROUND(100.0*COUNT(*)/? , 1) FROM registros_operativos WHERE Nivel_Riesgo='Alto'",
        (total,)
    ).fetchone()[0]
    turno_riesgo = conn.execute(
        "SELECT Turno FROM registros_operativos WHERE Nivel_Riesgo='Alto' GROUP BY Turno ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()[0]
    telar_incid = conn.execute(
        "SELECT Telar FROM registros_operativos WHERE Resultado_Inspeccion='Rechazado' GROUP BY Telar ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()[0]
    cliente_rec = conn.execute(
        "SELECT Cliente FROM registros_operativos WHERE Reclamo_Cliente=1 GROUP BY Cliente ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()[0]
    pct_rechazado = conn.execute(
        "SELECT ROUND(100.0*COUNT(*)/? , 1) FROM registros_operativos WHERE Resultado_Inspeccion='Rechazado'",
        (total,)
    ).fetchone()[0]
    conn.close()

    return {
        "total_registros":    total,
        "pct_nivel_alto":     pct_alto,
        "pct_rechazado":      pct_rechazado,
        "turno_mayor_riesgo": turno_riesgo,
        "telar_mayor_incid":  telar_incid,
        "cliente_mas_reclamos": cliente_rec,
        "periodo":            "Dataset completo CREDITEX",
    }


def get_filter_options():
    """Retorna los valores únicos para los filtros del dashboard."""
    simular_latencia(50, 120)
    conn = _get_conn()
    telares  = ["Todos"] + sorted([r[0] for r in conn.execute("SELECT DISTINCT Telar FROM registros_operativos ORDER BY Telar").fetchall()])
    turnos   = ["Todos"] + [r[0] for r in conn.execute("SELECT DISTINCT Turno FROM registros_operativos").fetchall()]
    clientes = ["Todos"] + sorted([r[0] for r in conn.execute("SELECT DISTINCT Cliente FROM registros_operativos").fetchall()])
    niveles  = ["Todos"] + [r[0] for r in conn.execute("SELECT DISTINCT Nivel_Riesgo FROM registros_operativos").fetchall()]
    conn.close()
    return {"telares": telares, "turnos": turnos, "clientes": clientes, "niveles": niveles}
