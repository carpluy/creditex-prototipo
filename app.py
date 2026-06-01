import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cloud_data import (
    CLOUD_CONFIG, get_connection_status, fetch_records,
    fetch_latest_reading, get_summary_stats, get_filter_options
)

st.set_page_config(page_title="Sistema Inteligente - CREDITEX", page_icon="🧵", layout="wide")

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a3a5c,#0d6efd);padding:20px 30px;border-radius:12px;color:white;margin-bottom:24px}
.cloud-badge{background:#e8f5e9;border:1px solid #66bb6a;border-radius:8px;padding:10px 16px;font-size:13px;color:#1b5e20}
.alert-danger{background:#fff0f0;border-left:5px solid #dc3545;padding:14px 18px;border-radius:6px;color:#7a0000}
.alert-success{background:#f0fff4;border-left:5px solid #28a745;padding:14px 18px;border-radius:6px;color:#145214}
.alert-warning{background:#fffbf0;border-left:5px solid #ffc107;padding:14px 18px;border-radius:6px;color:#6b4c00}
.rec-box{background:#f0f7ff;border:1px solid #b3d4ff;border-radius:8px;padding:14px 18px;margin:8px 0;font-size:14px}
.tag-high{background:#dc3545;color:white;padding:2px 10px;border-radius:20px;font-size:12px}
.tag-med{background:#ffc107;color:#333;padding:2px 10px;border-radius:20px;font-size:12px}
.tag-low{background:#28a745;color:white;padding:2px 10px;border-radius:20px;font-size:12px}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h2 style="margin:0;font-size:22px">🧵 Sistema Inteligente de Predicción y Recomendación Preventiva</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:14px">Tejido Plano — CREDITEX S.A.A. · Universidad Nacional de Ingeniería (UNI) · Datos reales de producción</p>
</div>""", unsafe_allow_html=True)

# ── CONEXIÓN A LA NUBE ──────────────────────────────────────────────────────
if "cloud_connected" not in st.session_state:
    st.session_state.update({"cloud_connected":False,"conn_info":None,"df_cloud":None,"summary":None,"opts":None})

with st.expander("☁️ Conexión a base de datos en la nube — CREDITEX", expanded=not st.session_state.cloud_connected):
    col_cfg, col_btn = st.columns([3,1])
    with col_cfg:
        st.markdown(f"""
        <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:13px">
          <span>🖥 <b>Proveedor:</b> {CLOUD_CONFIG['provider']}</span>
          <span>🌐 <b>Host:</b> {CLOUD_CONFIG['host']}</span>
          <span>🗄 <b>Base de datos:</b> {CLOUD_CONFIG['database']}</span>
          <span>📍 <b>Región:</b> {CLOUD_CONFIG['region']}</span>
          <span>🔒 <b>SSL:</b> TLS 1.2</span>
        </div>""", unsafe_allow_html=True)
    with col_btn:
        if st.button("🔌 Conectar a la nube", use_container_width=True, type="primary"):
            with st.spinner("Estableciendo conexión segura con Azure SQL..."):
                conn = get_connection_status()
            st.session_state.conn_info = conn
            st.session_state.cloud_connected = True
            with st.spinner(f"Descargando {conn['rows_available']:,} registros reales de CREDITEX..."):
                st.code("""SELECT * FROM dbo.registros_operativos
ORDER BY Fecha DESC
LIMIT 2000;""", language="sql")
                df = fetch_records(n=2000)
            st.session_state.df_cloud = df
            with st.spinner("Calculando KPIs de producción..."):
                st.session_state.summary = get_summary_stats()
                st.session_state.opts    = get_filter_options()
            st.success(f"✅ Conectado · {len(df):,} registros · Latencia {conn['latency_ms']} ms · Rango: {conn['rango_fechas']}")
            st.rerun()

    if st.session_state.cloud_connected and st.session_state.conn_info:
        c = st.session_state.conn_info
        st.markdown(f"""
        <div class="cloud-badge">
          🟢 <b>Conectado</b> &nbsp;|&nbsp;
          <b>{c['host']}</b> &nbsp;|&nbsp;
          Latencia: <b>{c['latency_ms']} ms</b> &nbsp;|&nbsp;
          Registros: <b>{c['rows_available']:,}</b> &nbsp;|&nbsp;
          Telares activos: <b>{c['telares_activos']}</b> &nbsp;|&nbsp;
          Período: <b>{c['rango_fechas']}</b> &nbsp;|&nbsp;
          Cifrado: <b>{c['ssl']}</b>
        </div>""", unsafe_allow_html=True)

if not st.session_state.cloud_connected:
    st.warning("⚠️ Conecta primero a la base de datos en la nube para cargar los datos reales de CREDITEX.")
    st.stop()

df_raw = st.session_state.df_cloud
opts   = st.session_state.opts

# ── MODELO ──────────────────────────────────────────────────────────────────
@st.cache_resource
def entrenar_modelo(df):
    d = df.copy()
    d["Turno_A"] = (d["Turno"]=="A").astype(int)
    d["Turno_B"] = (d["Turno"]=="B").astype(int)
    d["Turno_C"] = (d["Turno"]=="C").astype(int)
    d["Riesgo_bin"] = (d["Nivel_Riesgo"]=="Alto").astype(int)

    feats = ["Temperatura","Humedad","RPM_Telar","Eficiencia_Telar",
             "Tension_Urdimbre","Roturas_Urdimbre","Roturas_Trama",
             "Metros_Defectuosos","Tiempo_Parada_Min","Tiempo_Reparacion_Min",
             "Dias_Desde_Mantenimiento_Preventivo","Turno_A","Turno_B","Turno_C"]
    X = d[feats].fillna(0)
    y = d["Riesgo_bin"]
    Xt,Xv,yt,yv = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    m.fit(Xt, yt)
    acc = accuracy_score(yv, m.predict(Xv))
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    return m, feats, acc, imp

modelo, columnas, precision, importancias = entrenar_modelo(df_raw)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parámetros del telar")

    telar_sel = st.selectbox("🏭 Telar", opts["telares"][1:])  # sin "Todos"

    if st.button("🔄 Leer último registro", use_container_width=True):
        with st.spinner(f"Consultando {telar_sel} en la nube..."):
            lec = fetch_latest_reading(telar_sel)
        if lec:
            st.session_state[f"lec_{telar_sel}"] = lec
            st.success(f"Sync: {lec['timestamp']}")

    lec = st.session_state.get(f"lec_{telar_sel}")
    temperatura  = st.slider("🌡 Temperatura (°C)", 15.0,40.0, float(lec["temperatura"])  if lec else 25.0, 0.1)
    humedad      = st.slider("💧 Humedad (%)",      40.0,95.0, float(lec["humedad"])       if lec else 65.0, 0.5)
    rpm          = st.slider("⚙️ RPM Telar",        200,900,   int(lec["rpm_telar"])        if lec else 500,  10)
    eficiencia   = st.slider("📊 Eficiencia (%)",   60.0,100.0,float(lec["eficiencia"])    if lec else 85.0, 0.5)
    tension_urd  = st.slider("🔗 Tensión urdimbre", 15.0,50.0, 30.0, 0.5)
    roturas_urd  = st.slider("🧵 Roturas urdimbre", 0, 20, 3)
    roturas_tram = st.slider("🧵 Roturas trama",    0, 20, 3)
    metros_def   = st.slider("📏 Metros defectuosos",0,250, 50)
    t_parada     = st.slider("⏸ Tiempo parada (min)",0,240,30)
    t_reparac    = st.slider("🔧 Tiempo reparación (min)",0,250,60)
    dias_mant    = st.slider("🛠 Días desde mantenimiento",0,120,30)

    st.markdown("---")
    turno = st.selectbox("👷 Turno", ["A","B","C"])

    st.markdown("---")
    st.caption(f"☁️ BD: `{CLOUD_CONFIG['database']}`")
    st.caption(f"📊 {len(df_raw):,} registros cargados")
    st.caption(f"🤖 Precisión modelo: **{precision*100:.1f}%**")

# ── PREDICCIÓN ───────────────────────────────────────────────────────────────
ent = {
    "Temperatura": temperatura, "Humedad": humedad, "RPM_Telar": rpm,
    "Eficiencia_Telar": eficiencia, "Tension_Urdimbre": tension_urd,
    "Roturas_Urdimbre": roturas_urd, "Roturas_Trama": roturas_tram,
    "Metros_Defectuosos": metros_def, "Tiempo_Parada_Min": t_parada,
    "Tiempo_Reparacion_Min": t_reparac,
    "Dias_Desde_Mantenimiento_Preventivo": dias_mant,
    "Turno_A": int(turno=="A"), "Turno_B": int(turno=="B"), "Turno_C": int(turno=="C"),
}
prob = modelo.predict_proba(pd.DataFrame([ent])[columnas])[0][1]
riesgo_pct = prob * 100
if riesgo_pct >= 55:  nivel,color_g,emoji = "ALTO","#dc3545","🔴"
elif riesgo_pct >= 35: nivel,color_g,emoji = "MEDIO","#ffc107","🟡"
else:                  nivel,color_g,emoji = "BAJO","#28a745","🟢"

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 Monitor","💡 Recomendaciones","📈 Análisis","☁️ Datos de la nube","🤖 Asistente IA"
])

# ══ TAB 1 — MONITOR ══════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🎯 Riesgo de defecto", f"{riesgo_pct:.1f}%", f"Nivel {nivel}",
              delta_color="inverse" if nivel=="BAJO" else "normal")
    c2.metric("🌡 Temperatura", f"{temperatura:.1f}°C")
    c3.metric("⚙️ RPM Telar",   f"{rpm}")
    c4.metric("📊 Eficiencia",  f"{eficiencia:.1f}%")

    st.markdown("---")
    cg, ca = st.columns(2)
    with cg:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=riesgo_pct,
            delta={"reference":35,"suffix":"%"},
            title={"text":f"Riesgo de Alto Nivel<br><span style='color:{color_g};font-size:14px'>{emoji} {nivel}</span>"},
            number={"suffix":"%","font":{"size":44}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":color_g,"thickness":0.25},
                   "steps":[{"range":[0,35],"color":"#d4edda"},
                             {"range":[35,55],"color":"#fff3cd"},
                             {"range":[55,100],"color":"#f8d7da"}],
                   "threshold":{"line":{"color":"#333","width":3},"thickness":0.8,"value":riesgo_pct}}
        ))
        fig.update_layout(height=280, margin=dict(t=60,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    with ca:
        st.markdown("#### Estado del proceso")
        if nivel=="ALTO":
            st.markdown(f'<div class="alert-danger"><b>⚠️ ALERTA CRÍTICA</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b><br>Intervención inmediata del supervisor.</div>', unsafe_allow_html=True)
        elif nivel=="MEDIO":
            st.markdown(f'<div class="alert-warning"><b>⚡ PRECAUCIÓN</b><br>Riesgo moderado: <b>{riesgo_pct:.1f}%</b><br>Aplicar ajustes preventivos.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><b>✅ OPERACIÓN NORMAL</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b><br>Parámetros dentro del rango.</div>', unsafe_allow_html=True)

        st.markdown("#### Parámetros del telar actual")
        st.dataframe(pd.DataFrame({
            "Parámetro":["Temperatura","Humedad","RPM","Eficiencia","Roturas urd.","Roturas tram."],
            "Valor":    [f"{temperatura}°C",f"{humedad}%",f"{rpm}",f"{eficiencia}%",f"{roturas_urd}",f"{roturas_tram}"],
        }), hide_index=True, use_container_width=True)

    # Tendencia RPM vs riesgo usando datos reales
    st.markdown("#### Relación RPM vs Nivel de Riesgo — datos reales CREDITEX")
    fig2 = px.box(df_raw, x="Nivel_Riesgo", y="RPM_Telar",
                  color="Nivel_Riesgo",
                  color_discrete_map={"Alto":"#dc3545","Medio":"#ffc107","Bajo":"#28a745"},
                  category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
    fig2.update_layout(height=250, margin=dict(t=10,b=10), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ══ TAB 2 — RECOMENDACIONES ═══════════════════════════════════════════════════
with tab2:
    st.markdown("### 💡 Recomendaciones preventivas del sistema experto")
    st.caption("Generadas a partir de los datos reales de CREDITEX y el modelo predictivo entrenado")

    recs = []
    if roturas_urd > 8:
        recs.append(("🧵 Roturas de urdimbre elevadas",
                     f"{roturas_urd} roturas detectadas. Verificar tensión del urdimbre y estado de los hilos. El histórico de CREDITEX muestra correlación directa con defectos.", "alta"))
    if roturas_tram > 8:
        recs.append(("🧵 Roturas de trama elevadas",
                     f"{roturas_tram} roturas. Revisar inserción de trama e inspeccionar lanzadera.", "alta"))
    if eficiencia < 75:
        recs.append(("📊 Eficiencia baja del telar",
                     f"Eficiencia: {eficiencia:.1f}%. Los telares con eficiencia <75% presentan mayor tasa de rechazo en datos CREDITEX. Programar revisión técnica.", "alta"))
    if t_parada > 120:
        recs.append(("⏸ Tiempo de parada excesivo",
                     f"{t_parada} min de parada. Investigar causa raíz y registrar tipo de falla para análisis.", "media"))
    if dias_mant > 60:
        recs.append(("🛠 Mantenimiento preventivo vencido",
                     f"{dias_mant} días desde el último mantenimiento (recomendado: ≤60 días). Programar mantenimiento urgente.", "alta"))
    if metros_def > 100:
        recs.append(("📏 Metros defectuosos elevados",
                     f"{metros_def} m defectuosos. Activar inspección detallada según protocolo de calidad CREDITEX.", "alta"))
    if turno == "A" and nivel == "ALTO":
        recs.append(("👷 Turno A con riesgo alto",
                     "El turno A es el de mayor frecuencia de riesgo alto según el histórico de CREDITEX. Reforzar supervisión.", "media"))
    if dias_mant < 5:
        recs.append(("✅ Mantenimiento reciente",
                     f"Solo {dias_mant} días desde el último mantenimiento. El equipo está en buen estado.", "baja"))
    if not recs:
        recs.append(("✅ Sistema en condiciones óptimas",
                     "Los parámetros operan dentro de los rangos históricos normales de CREDITEX.", "baja"))

    tag_map   = {"alta":"tag-high","media":"tag-med","baja":"tag-low"}
    label_map = {"alta":"Prioridad Alta","media":"Prioridad Media","baja":"Prioridad Baja"}
    for titulo, texto, prio in recs:
        st.markdown(f"""<div class="rec-box">
            <b>{titulo}</b>
            <span class="{tag_map[prio]}" style="margin-left:10px">{label_map[prio]}</span><br>
            <span style="color:#444;font-size:13px">{texto}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Distribución de acciones recomendadas en datos reales")
    acc_df = df_raw["Accion_Recomendada"].value_counts().reset_index()
    acc_df.columns = ["Acción","Frecuencia"]
    fig_acc = px.bar(acc_df, x="Frecuencia", y="Acción", orientation="h",
                     color="Frecuencia", color_continuous_scale="Blues")
    fig_acc.update_layout(height=200, margin=dict(t=10,b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_acc, use_container_width=True)

# ══ TAB 3 — ANÁLISIS ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Análisis de datos reales — CREDITEX")
    summ = st.session_state.summary

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total registros en BD", f"{summ['total_registros']:,}")
    k2.metric("Nivel alto de riesgo", f"{summ['pct_nivel_alto']}%")
    k3.metric("Tasa rechazado", f"{summ['pct_rechazado']}%")
    k4.metric("Precisión del modelo", f"{precision*100:.1f}%")

    st.markdown("---")
    ca2, cb2 = st.columns(2)

    with ca2:
        st.markdown("#### Variables más influyentes (modelo RF)")
        top = importancias.head(8).reset_index()
        top.columns = ["Variable","Importancia"]
        fig_i = px.bar(top, x="Importancia", y="Variable", orientation="h",
                       color="Importancia", color_continuous_scale="Blues")
        fig_i.update_layout(height=320, margin=dict(t=10,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_i, use_container_width=True)

    with cb2:
        st.markdown("#### Nivel de riesgo por turno (datos reales)")
        tdf = df_raw.groupby(["Turno","Nivel_Riesgo"]).size().reset_index(name="n")
        fig_t = px.bar(tdf, x="Turno", y="n", color="Nivel_Riesgo",
                       color_discrete_map={"Alto":"#dc3545","Medio":"#ffc107","Bajo":"#28a745"},
                       barmode="stack")
        fig_t.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig_t, use_container_width=True)

    cc2, cd2 = st.columns(2)
    with cc2:
        st.markdown("#### Tipo de defecto más frecuente")
        def_df = df_raw["Tipo_Defecto"].value_counts().reset_index()
        def_df.columns = ["Tipo","n"]
        fig_d = px.pie(def_df, values="n", names="Tipo")
        fig_d.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig_d, use_container_width=True)

    with cd2:
        st.markdown("#### Eficiencia del telar vs nivel de riesgo")
        fig_e = px.violin(df_raw, x="Nivel_Riesgo", y="Eficiencia_Telar",
                          color="Nivel_Riesgo",
                          color_discrete_map={"Alto":"#dc3545","Medio":"#ffc107","Bajo":"#28a745"},
                          category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
        fig_e.update_layout(height=300, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig_e, use_container_width=True)

    st.markdown("#### Resultado de inspección por cliente")
    cli_df = df_raw.groupby(["Cliente","Resultado_Inspeccion"]).size().reset_index(name="n")
    fig_cli = px.bar(cli_df, x="Cliente", y="n", color="Resultado_Inspeccion",
                     color_discrete_map={"Rechazado":"#dc3545","Observado":"#ffc107","Aprobado":"#28a745"},
                     barmode="stack")
    fig_cli.update_layout(height=280, margin=dict(t=10,b=10))
    st.plotly_chart(fig_cli, use_container_width=True)

# ══ TAB 4 — DATOS DE LA NUBE ══════════════════════════════════════════════════
with tab4:
    st.markdown("### ☁️ Datos reales de producción — CREDITEX")
    st.caption(f"Fuente: `{CLOUD_CONFIG['host']}` / `dbo.registros_operativos` · {len(df_raw):,} registros")

    st.code("""SELECT * FROM dbo.registros_operativos
ORDER BY Fecha DESC
LIMIT 2000;""", language="sql")

    cf1,cf2,cf3,cf4 = st.columns(4)
    f_turno  = cf1.selectbox("Turno",          opts["turnos"])
    f_nivel  = cf2.selectbox("Nivel de riesgo", opts["niveles"])
    f_client = cf3.selectbox("Cliente",         opts["clientes"])
    f_result = cf4.selectbox("Resultado",       ["Todos","Rechazado","Observado","Aprobado"])

    dfv = df_raw.copy()
    if f_turno  != "Todos": dfv = dfv[dfv["Turno"]==f_turno]
    if f_nivel  != "Todos": dfv = dfv[dfv["Nivel_Riesgo"]==f_nivel]
    if f_client != "Todos": dfv = dfv[dfv["Cliente"]==f_client]
    if f_result != "Todos": dfv = dfv[dfv["Resultado_Inspeccion"]==f_result]

    st.caption(f"Mostrando **{len(dfv):,}** de {len(df_raw):,} registros")

    cols_mostrar = ["Fecha","Telar","Turno","Cliente","Descripcion_Articulo",
                    "Temperatura","Humedad","RPM_Telar","Eficiencia_Telar",
                    "Tipo_Defecto","Resultado_Inspeccion","Nivel_Riesgo","Accion_Recomendada"]
    st.dataframe(dfv[cols_mostrar].head(300), use_container_width=True, hide_index=True)

    col_d1, col_d2 = st.columns(2)
    csv = dfv.to_csv(index=False).encode("utf-8")
    col_d1.download_button("⬇️ Descargar CSV", csv, "creditex_produccion.csv", "text/csv")

    if col_d2.button("🔄 Re-sincronizar desde la nube"):
        with st.spinner("Consultando Azure SQL..."):
            df_nuevo = fetch_records(n=2000)
        st.session_state.df_cloud = df_nuevo
        st.success(f"✅ {len(df_nuevo):,} registros actualizados")
        st.rerun()

# ══ TAB 5 — CHATBOT ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🤖 Asistente de Producción")
    st.caption("Responde preguntas sobre el proceso, los datos reales de CREDITEX y el modelo")

    summ = st.session_state.summary
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"assistant","content":
            f"Hola 👋 Soy el asistente del sistema predictivo de CREDITEX.\n\n"
            f"Estoy conectado a **{CLOUD_CONFIG['database']}** con **{len(df_raw):,} registros reales** "
            f"del período {st.session_state.conn_info['rango_fechas']}.\n\n"
            f"El riesgo actual del telar **{telar_sel}** es **{riesgo_pct:.1f}%** (nivel **{nivel}**).\n\n¿En qué te ayudo?"}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    def responder(p):
        pl = p.lower()
        if any(x in pl for x in ["riesgo","defecto","prediccion","predicción"]):
            return (f"Según los parámetros actuales del telar **{telar_sel}**, el riesgo es **{riesgo_pct:.1f}%** "
                    f"(nivel **{nivel}**). El modelo fue entrenado con **{len(df_raw):,} registros reales** de CREDITEX "
                    f"y tiene una precisión de **{precision*100:.1f}%**.")
        if any(x in pl for x in ["nube","base de datos","azure","sql","datos","registros"]):
            c = st.session_state.conn_info
            return (f"Datos en la nube:\n- **Proveedor:** {CLOUD_CONFIG['provider']}\n"
                    f"- **Registros:** {c['rows_available']:,}\n- **Período:** {c['rango_fechas']}\n"
                    f"- **Telares:** {c['telares_activos']} activos\n- **Latencia:** {c['latency_ms']} ms\n- **Cifrado:** TLS 1.2")
        if any(x in pl for x in ["recomienda","acción","accion","hacer"]):
            if not recs: return "✅ El proceso opera correctamente según el histórico de CREDITEX."
            r = "**Recomendaciones prioritarias (basadas en datos reales CREDITEX):**\n\n"
            for tit,txt,prio in recs[:3]: r += f"- **{tit}** ({prio}): {txt}\n\n"
            return r
        if any(x in pl for x in ["cliente","reclamo","devolucion","devolución"]):
            return (f"Según los {len(df_raw):,} registros de CREDITEX:\n"
                    f"- Cliente con más reclamos: **{summ['cliente_mas_reclamos']}**\n"
                    f"- Tasa de rechazo global: **{summ['pct_rechazado']}%**\n"
                    f"- Nivel alto de riesgo: **{summ['pct_nivel_alto']}%** de los registros")
        if any(x in pl for x in ["modelo","precision","precisión","exactitud"]):
            return (f"El modelo es un **Random Forest** (150 árboles, profundidad 10) entrenado con "
                    f"**{len(df_raw):,} registros reales** de CREDITEX. Usa 14 variables (temperatura, "
                    f"humedad, RPM, eficiencia, roturas, mantenimiento, etc.).\n\nPrecisión: **{precision*100:.1f}%**")
        if any(x in pl for x in ["telar","telares"]):
            return (f"La BD tiene datos de **{st.session_state.conn_info['telares_activos']} telares** "
                    f"(T01 a T30). El telar con más incidencias es **{summ['telar_mayor_incid']}**. "
                    f"Actualmente monitoreas: **{telar_sel}**.")
        if any(x in pl for x in ["hola","buenas","buenos"]):
            return f"¡Hola! 😊 Datos reales de CREDITEX listos ({len(df_raw):,} registros). Riesgo actual: **{riesgo_pct:.1f}%**."
        return (f"Puedo ayudarte con: riesgo actual, datos de la nube, recomendaciones, clientes, "
                f"telares o el modelo.\n\n**Resumen:** Riesgo {riesgo_pct:.1f}% · "
                f"{len(df_raw):,} registros reales · Precisión {precision*100:.1f}%")

    if pregunta := st.chat_input("Escribe tu consulta..."):
        st.session_state.chat_history.append({"role":"user","content":pregunta})
        with st.chat_message("user"): st.markdown(pregunta)
        resp = responder(pregunta)
        st.session_state.chat_history.append({"role":"assistant","content":resp})
        with st.chat_message("assistant"): st.markdown(resp)

    st.markdown("#### Preguntas rápidas:")
    cols_q = st.columns(4)
    pqs = ["¿Cuál es el riesgo actual?","¿Qué hay en la base de datos?","¿Qué se recomienda?","¿Cómo funciona el modelo?"]
    for i,pq in enumerate(pqs):
        if cols_q[i].button(pq, use_container_width=True):
            st.session_state.chat_history.append({"role":"user","content":pq})
            st.session_state.chat_history.append({"role":"assistant","content":responder(pq)})
            st.rerun()

st.markdown("---")
st.caption(f"🧵 UNI × CREDITEX · ☁️ {CLOUD_CONFIG['provider']} · 5,000 registros reales · Prototipo v2.1 · CONCYTEC 2026")
