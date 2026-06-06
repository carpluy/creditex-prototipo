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
 
# Logo URL de CREDITEX (desde su web oficial)
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Creditex_logo.svg/320px-Creditex_logo.svg.png"
# Colores corporativos CREDITEX: azul marino oscuro + dorado
C_NAVY  = "#0a1f3c"
C_BLUE  = "#1a3f6f"
C_GOLD  = "#c8a84b"
C_LIGHT = "#e8f0fb"
 
st.markdown(f"""
<style>
/* ── FONDO GENERAL ── */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(160deg, #0a1525 0%, #0d2144 50%, #0a1525 100%);
    min-height: 100vh;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a1f3c 0%, #0d2a50 100%) !important;
    border-right: 2px solid {C_GOLD};
}}
[data-testid="stSidebar"] * {{ color: #e0eaff !important; }}
[data-testid="stSidebar"] .stSlider label {{ color: {C_GOLD} !important; font-weight:600; }}
 
/* ── HEADER PRINCIPAL ── */
.main-header {{
    background: linear-gradient(135deg, {C_NAVY} 0%, {C_BLUE} 60%, #1e5090 100%);
    border: 1px solid {C_GOLD};
    border-left: 6px solid {C_GOLD};
    padding: 20px 32px;
    border-radius: 14px;
    color: white;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}}
.header-logo {{
    background: white;
    border-radius: 10px;
    padding: 8px 14px;
    flex-shrink: 0;
}}
.header-title {{ font-size: 22px; font-weight: 700; color: white; margin: 0; }}
.header-sub {{ font-size: 13px; color: {C_GOLD}; margin: 4px 0 0; }}
 
/* ── TARJETAS MÉTRICAS ── */
[data-testid="metric-container"] {{
    background: linear-gradient(135deg, #0d2144, #163560) !important;
    border: 1px solid {C_GOLD}55;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}}
[data-testid="metric-container"] label {{ color: {C_GOLD} !important; font-size: 13px !important; font-weight: 600 !important; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: white !important; font-size: 28px !important; font-weight: 700 !important; }}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{ font-size: 12px !important; }}
 
/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {C_NAVY};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {C_GOLD}44;
    gap: 4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent !important;
    color: #8ab0d8 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: linear-gradient(135deg, {C_BLUE}, #1e5090) !important;
    color: {C_GOLD} !important;
    border: 1px solid {C_GOLD}66 !important;
}}
 
/* ── EXPANDER ── */
[data-testid="stExpander"] {{
    background: linear-gradient(135deg, #0d2144, #0f2a50) !important;
    border: 1px solid {C_GOLD}55 !important;
    border-radius: 12px !important;
}}
 
/* ── SELECTBOX / SLIDER ── */
[data-testid="stSelectbox"] > div > div {{
    background: #0d2a50 !important;
    border: 1px solid {C_GOLD}66 !important;
    color: white !important;
    border-radius: 8px !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: {C_GOLD} !important;
    border: 2px solid white !important;
}}
 
/* ── ALERTAS ── */
.alert-danger  {{ background:#2a0a0a;border-left:5px solid #ff4444;padding:14px 18px;border-radius:8px;color:#ffaaaa; }}
.alert-success {{ background:#0a2a14;border-left:5px solid #00cc66;padding:14px 18px;border-radius:8px;color:#aaffcc; }}
.alert-warning {{ background:#2a1f00;border-left:5px solid {C_GOLD};padding:14px 18px;border-radius:8px;color:#ffe08a; }}
.cloud-badge   {{ background:#0a2a14;border:1px solid #00cc66;border-radius:8px;padding:10px 16px;font-size:13px;color:#aaffcc; }}
 
/* ── RECOMENDACIONES ── */
.rec-box {{ background:linear-gradient(135deg,#0d2144,#122d55);border:1px solid {C_GOLD}44;border-radius:10px;padding:14px 18px;margin:8px 0;font-size:14px;color:#d0e4ff; }}
.tag-high {{ background:#cc2233;color:white;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600; }}
.tag-med  {{ background:{C_GOLD};color:#111;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600; }}
.tag-low  {{ background:#00994d;color:white;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600; }}
 
/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; border: 1px solid {C_GOLD}44; }}
 
/* ── BOTONES ── */
.stButton > button {{
    background: linear-gradient(135deg, {C_BLUE}, #1e5090) !important;
    color: {C_GOLD} !important;
    border: 1px solid {C_GOLD} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {C_GOLD}, #e0b860) !important;
    color: {C_NAVY} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(200,168,75,0.4) !important;
}}
 
/* ── CHAT ── */
[data-testid="stChatMessage"] {{
    background: #0d2144 !important;
    border: 1px solid {C_GOLD}33 !important;
    border-radius: 10px !important;
}}
[data-testid="stChatInput"] textarea {{
    background: #0d2144 !important;
    border: 1px solid {C_GOLD} !important;
    color: white !important;
    border-radius: 8px !important;
}}
 
/* ── TEXTO GENERAL ── */
h1,h2,h3,h4,p,span,label,div {{ color: #d0e4ff; }}
.stMarkdown p {{ color: #c0d8f0; }}
code {{ background: #0d2144 !important; color: {C_GOLD} !important; }}
 
/* ── SEPARADORES ── */
hr {{ border-color: {C_GOLD}44 !important; }}
 
/* ── SCROLL BAR ── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {C_NAVY}; }}
::-webkit-scrollbar-thumb {{ background: {C_GOLD}88; border-radius: 3px; }}
</style>""", unsafe_allow_html=True)
 
# ── HEADER con logo CREDITEX ────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <div class="header-logo">
    <img src="https://www.creditex.com.pe/wp-content/uploads/2021/03/logo-creditex.png"
         onerror="this.style.display='none';this.nextSibling.style.display='block'"
         height="52" style="display:block">
    <div style="display:none;font-weight:700;color:{C_NAVY};font-size:16px;letter-spacing:1px">CREDITEX</div>
  </div>
  <div>
    <div class="header-title">🧵 Sistema Inteligente de Predicción y Recomendación Preventiva</div>
    <div class="header-sub">Tejido Plano · CREDITEX S.A.A. · Universidad Nacional de Ingeniería (UNI) · Datos reales de producción · CONCYTEC 2026</div>
  </div>
</div>""", unsafe_allow_html=True)
 
# ── CONEXIÓN A LA NUBE ──────────────────────────────────────────────────────
if "cloud_connected" not in st.session_state:
    st.session_state.update({"cloud_connected":False,"conn_info":None,"df_cloud":None,"summary":None,"opts":None})
 
with st.expander("☁️ Conexión a base de datos en la nube — CREDITEX", expanded=not st.session_state.cloud_connected):
    col_cfg, col_btn = st.columns([3,1])
    with col_cfg:
        st.markdown(f"""
        <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:13px;color:#8ab0d8">
          <span>🖥 <b style="color:{C_GOLD}">Proveedor:</b> {CLOUD_CONFIG['provider']}</span>
          <span>🌐 <b style="color:{C_GOLD}">Host:</b> {CLOUD_CONFIG['host']}</span>
          <span>🗄 <b style="color:{C_GOLD}">Base de datos:</b> {CLOUD_CONFIG['database']}</span>
          <span>📍 <b style="color:{C_GOLD}">Región:</b> {CLOUD_CONFIG['region']}</span>
          <span>🔒 <b style="color:{C_GOLD}">SSL:</b> TLS 1.2</span>
        </div>""", unsafe_allow_html=True)
    with col_btn:
        if st.button("🔌 Conectar a la nube", use_container_width=True, type="primary"):
            with st.spinner("Estableciendo conexión segura con Azure SQL..."):
                conn = get_connection_status()
            st.session_state.conn_info = conn
            st.session_state.cloud_connected = True
            with st.spinner(f"Descargando {conn['rows_available']:,} registros reales de CREDITEX..."):
                st.code("""SELECT * FROM dbo.registros_operativos ORDER BY Fecha DESC LIMIT 2000;""", language="sql")
                df = fetch_records(n=2000)
            st.session_state.df_cloud = df
            with st.spinner("Calculando KPIs de producción..."):
                st.session_state.summary = get_summary_stats()
                st.session_state.opts    = get_filter_options()
            st.success(f"✅ Conectado · {len(df):,} registros · Latencia {conn['latency_ms']} ms · {conn['rango_fechas']}")
            st.rerun()
 
    if st.session_state.cloud_connected and st.session_state.conn_info:
        c = st.session_state.conn_info
        st.markdown(f"""
        <div class="cloud-badge">
          🟢 <b>Conectado</b> &nbsp;|&nbsp; <b>{c['host']}</b> &nbsp;|&nbsp;
          Latencia: <b>{c['latency_ms']} ms</b> &nbsp;|&nbsp;
          Registros: <b>{c['rows_available']:,}</b> &nbsp;|&nbsp;
          Telares: <b>{c['telares_activos']}</b> &nbsp;|&nbsp;
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
    st.markdown(f"""<div style="text-align:center;padding:12px 0;border-bottom:1px solid {C_GOLD}44;margin-bottom:16px">
        <img src="https://www.creditex.com.pe/wp-content/uploads/2021/03/logo-creditex.png"
             height="40" style="background:white;padding:4px 10px;border-radius:8px"
             onerror="this.style.display='none'">
        <div style="font-size:11px;color:{C_GOLD};margin-top:6px">Sistema Inteligente de Producción</div>
    </div>""", unsafe_allow_html=True)
 
    st.markdown(f"### ⚙️ Parámetros del telar")
    telar_sel = st.selectbox("🏭 Telar", opts["telares"][1:])
 
    if st.button("🔄 Leer último registro", use_container_width=True):
        with st.spinner(f"Consultando {telar_sel}..."):
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
    turno        = st.selectbox("👷 Turno", ["A","B","C"])
 
    st.markdown("---")
    st.markdown(f"<div style='font-size:11px;color:{C_GOLD}'>☁️ BD: <b>{CLOUD_CONFIG['database']}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#8ab0d8'>📊 {len(df_raw):,} registros cargados</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#8ab0d8'>🤖 Precisión modelo: <b style='color:{C_GOLD}'>{precision*100:.1f}%</b></div>", unsafe_allow_html=True)
 
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
if riesgo_pct >= 55:  nivel,color_g,emoji,color_text = "ALTO","#ff4444","🔴","#ffaaaa"
elif riesgo_pct >= 35: nivel,color_g,emoji,color_text = "MEDIO",C_GOLD,"🟡","#ffe08a"
else:                  nivel,color_g,emoji,color_text = "BAJO","#00cc66","🟢","#aaffcc"
 
# ── TABS ─────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 Monitor","💡 Recomendaciones","📈 Análisis","☁️ Datos de la nube","🤖 Asistente IA"
])
 
COLORS = [C_GOLD, "#4e8fdf", "#00cc66", "#ff6644", "#aa66ff", "#00cccc"]
 
# ══ TAB 1 — MONITOR ══════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🎯 Riesgo de defecto", f"{riesgo_pct:.1f}%", f"{emoji} Nivel {nivel}",
              delta_color="inverse" if nivel=="BAJO" else "normal")
    c2.metric("🌡 Temperatura", f"{temperatura:.1f}°C")
    c3.metric("⚙️ RPM Telar",   f"{rpm}")
    c4.metric("📊 Eficiencia",  f"{eficiencia:.1f}%")
 
    st.markdown("---")
    cg, ca = st.columns([1,1])
 
    with cg:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=riesgo_pct,
            delta={"reference":35,"suffix":"%","increasing":{"color":"#ff4444"},"decreasing":{"color":"#00cc66"}},
            title={"text":f"<b>Riesgo de Defecto</b><br><span style='font-size:15px;color:{color_g}'>{emoji} Nivel {nivel}</span>",
                   "font":{"color":"white"}},
            number={"suffix":"%","font":{"size":52,"color":"white"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#8ab0d8","tickfont":{"color":"#8ab0d8"}},
                "bar":{"color":color_g,"thickness":0.28},
                "bgcolor":"#0d2144",
                "bordercolor":C_GOLD,"borderwidth":1,
                "steps":[{"range":[0,35],"color":"#0a2a14"},
                         {"range":[35,55],"color":"#2a1f00"},
                         {"range":[55,100],"color":"#2a0a0a"}],
                "threshold":{"line":{"color":C_GOLD,"width":3},"thickness":0.85,"value":riesgo_pct},
            }
        ))
        fig.update_layout(height=300, margin=dict(t=70,b=20,l=20,r=20),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color":"white"})
        st.plotly_chart(fig, use_container_width=True)
 
    with ca:
        st.markdown("#### Estado del proceso")
        if nivel=="ALTO":
            st.markdown(f'<div class="alert-danger"><b>⚠️ ALERTA CRÍTICA</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b><br>Intervención inmediata del supervisor de turno.</div>', unsafe_allow_html=True)
        elif nivel=="MEDIO":
            st.markdown(f'<div class="alert-warning"><b>⚡ PRECAUCIÓN</b><br>Riesgo moderado: <b>{riesgo_pct:.1f}%</b><br>Aplicar ajustes preventivos.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><b>✅ OPERACIÓN NORMAL</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b><br>Parámetros dentro del rango.</div>', unsafe_allow_html=True)
 
        st.markdown("#### Parámetros del telar actual")
        st.dataframe(pd.DataFrame({
            "Parámetro":["Temperatura","Humedad","RPM","Eficiencia","Roturas urd.","Roturas tram."],
            "Valor":    [f"{temperatura}°C",f"{humedad}%",f"{rpm}",f"{eficiencia}%",f"{roturas_urd}",f"{roturas_tram}"],
        }), hide_index=True, use_container_width=True)
 
    st.markdown("#### Relación RPM vs Nivel de Riesgo — datos reales CREDITEX")
    fig2 = px.box(df_raw, x="Nivel_Riesgo", y="RPM_Telar", color="Nivel_Riesgo",
                  color_discrete_map={"Alto":"#ff4444","Medio":C_GOLD,"Bajo":"#00cc66"},
                  category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
    fig2.update_layout(height=260, margin=dict(t=10,b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,33,68,0.6)",
                       showlegend=False, font={"color":"#d0e4ff"},
                       xaxis={"gridcolor":"#1a3f6f"},yaxis={"gridcolor":"#1a3f6f"})
    st.plotly_chart(fig2, use_container_width=True)
 
# ══ TAB 2 — RECOMENDACIONES ═══════════════════════════════════════════════════
with tab2:
    st.markdown("### 💡 Recomendaciones preventivas del sistema experto")
    recs = []
    if roturas_urd > 8:
        recs.append(("🧵 Roturas de urdimbre elevadas", f"{roturas_urd} roturas. Verificar tensión y estado de hilos.", "alta"))
    if roturas_tram > 8:
        recs.append(("🧵 Roturas de trama elevadas", f"{roturas_tram} roturas. Revisar lanzadera.", "alta"))
    if eficiencia < 75:
        recs.append(("📊 Eficiencia baja", f"{eficiencia:.1f}%. Los telares con eficiencia <75% presentan mayor tasa de rechazo.", "alta"))
    if t_parada > 120:
        recs.append(("⏸ Tiempo de parada excesivo", f"{t_parada} min. Investigar causa raíz.", "media"))
    if dias_mant > 60:
        recs.append(("🛠 Mantenimiento preventivo vencido", f"{dias_mant} días desde último mantenimiento. Programar urgente.", "alta"))
    if metros_def > 100:
        recs.append(("📏 Metros defectuosos elevados", f"{metros_def} m. Activar inspección detallada.", "alta"))
    if turno == "A" and nivel == "ALTO":
        recs.append(("👷 Turno A con riesgo alto", "Turno de mayor frecuencia de riesgo alto. Reforzar supervisión.", "media"))
    if not recs:
        recs.append(("✅ Sistema en condiciones óptimas", "Parámetros dentro de los rangos históricos normales de CREDITEX.", "baja"))
 
    tag_map   = {"alta":"tag-high","media":"tag-med","baja":"tag-low"}
    label_map = {"alta":"Prioridad Alta","media":"Prioridad Media","baja":"Prioridad Baja"}
    for titulo, texto, prio in recs:
        st.markdown(f"""<div class="rec-box">
            <b style="color:white">{titulo}</b>
            <span class="{tag_map[prio]}" style="margin-left:10px">{label_map[prio]}</span><br>
            <span style="font-size:13px">{texto}</span>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("#### Distribución de acciones recomendadas en datos reales")
    acc_df = df_raw["Accion_Recomendada"].value_counts().reset_index()
    acc_df.columns = ["Acción","Frecuencia"]
    fig_acc = px.bar(acc_df, x="Frecuencia", y="Acción", orientation="h",
                     color="Frecuencia", color_continuous_scale=[[0,C_BLUE],[1,C_GOLD]])
    fig_acc.update_layout(height=220, margin=dict(t=10,b=10), coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,33,68,0.6)",
                          font={"color":"#d0e4ff"}, xaxis={"gridcolor":"#1a3f6f"}, yaxis={"gridcolor":"#1a3f6f"})
    st.plotly_chart(fig_acc, use_container_width=True)
 
# ══ TAB 3 — ANÁLISIS ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Análisis de datos reales — CREDITEX")
    summ = st.session_state.summary
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total registros en BD", f"{summ['total_registros']:,}")
    k2.metric("Nivel alto de riesgo",  f"{summ['pct_nivel_alto']}%")
    k3.metric("Tasa rechazado",        f"{summ['pct_rechazado']}%")
    k4.metric("Precisión del modelo",  f"{precision*100:.1f}%")
 
    st.markdown("---")
    ca2,cb2 = st.columns(2)
    plt_cfg = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,33,68,0.6)",
                   font={"color":"#d0e4ff"}, margin=dict(t=10,b=10),
                   xaxis={"gridcolor":"#1a3f6f"}, yaxis={"gridcolor":"#1a3f6f"})
 
    with ca2:
        st.markdown("#### Variables más influyentes (modelo RF)")
        top = importancias.head(8).reset_index()
        top.columns = ["Variable","Importancia"]
        fig_i = px.bar(top, x="Importancia", y="Variable", orientation="h",
                       color="Importancia", color_continuous_scale=[[0,C_BLUE],[1,C_GOLD]])
        fig_i.update_layout(height=320, coloraxis_showscale=False, **plt_cfg)
        st.plotly_chart(fig_i, use_container_width=True)
 
    with cb2:
        st.markdown("#### Nivel de riesgo por turno")
        tdf = df_raw.groupby(["Turno","Nivel_Riesgo"]).size().reset_index(name="n")
        fig_t = px.bar(tdf, x="Turno", y="n", color="Nivel_Riesgo",
                       color_discrete_map={"Alto":"#ff4444","Medio":C_GOLD,"Bajo":"#00cc66"},
                       barmode="stack")
        fig_t.update_layout(height=320, **plt_cfg)
        st.plotly_chart(fig_t, use_container_width=True)
 
    cc2,cd2 = st.columns(2)
    with cc2:
        st.markdown("#### Tipo de defecto más frecuente")
        def_df = df_raw["Tipo_Defecto"].value_counts().reset_index()
        def_df.columns = ["Tipo","n"]
        fig_d = px.pie(def_df, values="n", names="Tipo",
                       color_discrete_sequence=[C_GOLD,C_BLUE,"#00cc66","#ff4444","#aa66ff","#00cccc"])
        fig_d.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={"color":"#d0e4ff"}, margin=dict(t=10,b=10))
        st.plotly_chart(fig_d, use_container_width=True)
 
    with cd2:
        st.markdown("#### Eficiencia del telar vs nivel de riesgo")
        fig_e = px.violin(df_raw, x="Nivel_Riesgo", y="Eficiencia_Telar", color="Nivel_Riesgo",
                          color_discrete_map={"Alto":"#ff4444","Medio":C_GOLD,"Bajo":"#00cc66"},
                          category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
        fig_e.update_layout(height=300, showlegend=False, **plt_cfg)
        st.plotly_chart(fig_e, use_container_width=True)
 
    st.markdown("#### Resultado de inspección por cliente")
    cli_df = df_raw.groupby(["Cliente","Resultado_Inspeccion"]).size().reset_index(name="n")
    fig_cli = px.bar(cli_df, x="Cliente", y="n", color="Resultado_Inspeccion",
                     color_discrete_map={"Rechazado":"#ff4444","Observado":C_GOLD,"Aprobado":"#00cc66"},
                     barmode="stack")
    fig_cli.update_layout(height=280, **plt_cfg)
    st.plotly_chart(fig_cli, use_container_width=True)
 
# ══ TAB 4 — DATOS DE LA NUBE ══════════════════════════════════════════════════
with tab4:
    st.markdown("### ☁️ Datos reales de producción — CREDITEX")
    st.code("""SELECT * FROM dbo.registros_operativos ORDER BY Fecha DESC LIMIT 2000;""", language="sql")
 
    cf1,cf2,cf3,cf4 = st.columns(4)
    f_turno  = cf1.selectbox("Turno",           opts["turnos"])
    f_nivel  = cf2.selectbox("Nivel de riesgo",  opts["niveles"])
    f_client = cf3.selectbox("Cliente",          opts["clientes"])
    f_result = cf4.selectbox("Resultado",        ["Todos","Rechazado","Observado","Aprobado"])
 
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
 
    col_d1,col_d2 = st.columns(2)
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
    st.markdown("### 🤖 Asistente de Producción CREDITEX")
    summ = st.session_state.summary
 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"assistant","content":
            f"Hola 👋 Soy el asistente del sistema predictivo de CREDITEX.\n\n"
            f"Conectado a **{CLOUD_CONFIG['database']}** con **{len(df_raw):,} registros reales**.\n\n"
            f"Riesgo actual del telar **{telar_sel}**: **{riesgo_pct:.1f}%** (nivel **{nivel}**).\n\n¿En qué te ayudo?"}]
 
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
 
    def responder(p):
        pl = p.lower()
        if any(x in pl for x in ["riesgo","defecto","prediccion","predicción"]):
            return f"Riesgo actual del telar **{telar_sel}**: **{riesgo_pct:.1f}%** (nivel **{nivel}**). Modelo entrenado con **{len(df_raw):,} registros reales** — precisión **{precision*100:.1f}%**."
        if any(x in pl for x in ["nube","base de datos","azure","sql","datos","registros"]):
            c = st.session_state.conn_info
            return f"- **Proveedor:** {CLOUD_CONFIG['provider']}\n- **Registros:** {c['rows_available']:,}\n- **Período:** {c['rango_fechas']}\n- **Telares:** {c['telares_activos']} activos\n- **Latencia:** {c['latency_ms']} ms"
        if any(x in pl for x in ["recomienda","acción","accion","hacer"]):
            if not recs: return "✅ El proceso opera correctamente según el histórico de CREDITEX."
            r = "**Recomendaciones prioritarias:**\n\n"
            for tit,txt,prio in recs[:3]: r += f"- **{tit}** ({prio}): {txt}\n\n"
            return r
        if any(x in pl for x in ["cliente","reclamo"]):
            return f"- Cliente con más reclamos: **{summ['cliente_mas_reclamos']}**\n- Tasa de rechazo global: **{summ['pct_rechazado']}%**\n- Nivel alto de riesgo: **{summ['pct_nivel_alto']}%**"
        if any(x in pl for x in ["modelo","precision","precisión"]):
            return f"**Random Forest** (150 árboles) entrenado con **{len(df_raw):,} registros reales** de CREDITEX. Precisión: **{precision*100:.1f}%**"
        if any(x in pl for x in ["hola","buenas"]):
            return f"¡Hola! 😊 {len(df_raw):,} registros de CREDITEX listos. Riesgo actual: **{riesgo_pct:.1f}%**."
        return f"Puedo ayudarte con: riesgo, nube, recomendaciones, clientes o el modelo.\n\n**Resumen:** Riesgo {riesgo_pct:.1f}% · {len(df_raw):,} registros · Precisión {precision*100:.1f}%"
 
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
st.markdown(f"<div style='text-align:center;font-size:11px;color:{C_GOLD}88'>🧵 CREDITEX S.A.A. · Universidad Nacional de Ingeniería (UNI) · ☁️ Azure SQL Database · CONCYTEC 2026 · Prototipo v3.0</div>", unsafe_allow_html=True)
 
