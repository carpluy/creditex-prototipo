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
 
st.set_page_config(
    page_title="Sistema Inteligente - CREDITEX",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
C_NAVY  = "#0a1f3c"
C_BLUE  = "#1a3f6f"
C_GOLD  = "#c8a84b"
C_DARK  = "#060f1e"
 
st.markdown(f"""
<style>
/* FONDO */
.stApp {{ background: {C_DARK} !important; }}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,#071528,#0d2244) !important;
    border-right: 2px solid {C_GOLD} !important;
    min-width: 260px !important;
}}
section[data-testid="stSidebar"] * {{ color: #cce0ff !important; }}
section[data-testid="stSidebar"] h3 {{ color: {C_GOLD} !important; font-size:16px !important; }}
 
/* HEADER */
.main-header {{
    background: linear-gradient(135deg,#071528 0%,#0f2d55 60%,#1a3f6f 100%);
    border: 1px solid {C_GOLD};
    border-left: 7px solid {C_GOLD};
    padding: 22px 36px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}}
.header-title {{ font-size:26px; font-weight:800; color:white; margin:0; letter-spacing:.5px; }}
.header-sub   {{ font-size:14px; color:{C_GOLD}; margin:6px 0 0; }}
 
/* MÉTRICAS */
[data-testid="metric-container"] {{
    background: linear-gradient(135deg,#0c2040,#122d55) !important;
    border: 1px solid {C_GOLD}66 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}}
[data-testid="stMetricLabel"] {{ color:{C_GOLD} !important; font-size:14px !important; font-weight:700 !important; }}
[data-testid="stMetricValue"] {{ color:white !important; font-size:36px !important; font-weight:800 !important; }}
[data-testid="stMetricDelta"] {{ font-size:13px !important; font-weight:600 !important; }}
 
/* TABS */
[data-baseweb="tab-list"] {{
    background: #0c2040 !important;
    border-radius: 12px !important;
    padding: 6px !important;
    border: 1px solid {C_GOLD}44 !important;
    gap: 4px !important;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    color: #7aaad8 !important;
    border-radius: 8px !important;
    font-size:15px !important;
    font-weight:600 !important;
    padding: 10px 22px !important;
    border: none !important;
}}
[aria-selected="true"] {{
    background: linear-gradient(135deg,{C_BLUE},{C_GOLD}88) !important;
    color: white !important;
}}
 
/* EXPANDER */
[data-testid="stExpander"] {{
    background: #0c2040 !important;
    border: 1px solid {C_GOLD}55 !important;
    border-radius: 12px !important;
}}
[data-testid="stExpander"] summary {{ color:{C_GOLD} !important; font-weight:700 !important; font-size:15px !important; }}
 
/* ALERTAS */
.alert-danger  {{ background:#1a0505;border-left:6px solid #ff3333;padding:16px 20px;border-radius:10px;color:#ffbbbb;font-size:15px; }}
.alert-success {{ background:#031a0a;border-left:6px solid #00cc55;padding:16px 20px;border-radius:10px;color:#aaffcc;font-size:15px; }}
.alert-warning {{ background:#1a1000;border-left:6px solid {C_GOLD};padding:16px 20px;border-radius:10px;color:#ffe08a;font-size:15px; }}
.cloud-badge   {{ background:#031a0a;border:1px solid #00cc55;border-radius:10px;padding:12px 18px;font-size:14px;color:#aaffcc; }}
 
/* RECOMENDACIONES */
.rec-box {{
    background: linear-gradient(135deg,#0c2040,#102848);
    border: 1px solid {C_GOLD}55;
    border-left: 4px solid {C_GOLD};
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 15px;
    color: #cce0ff;
}}
.tag-high {{ background:#cc2233;color:white;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }}
.tag-med  {{ background:{C_GOLD};color:#111;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }}
.tag-low  {{ background:#009944;color:white;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }}
 
/* TABLA — forzar fondo oscuro */
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; border:1px solid {C_GOLD}44; }}
iframe {{ background: #0c2040 !important; }}
 
/* BOTONES */
.stButton>button {{
    background: linear-gradient(135deg,{C_BLUE},{C_GOLD}aa) !important;
    color: white !important;
    border: 1px solid {C_GOLD} !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all .2s !important;
}}
.stButton>button:hover {{
    background: linear-gradient(135deg,{C_GOLD},{C_BLUE}) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(200,168,75,.5) !important;
}}
 
/* CHAT */
[data-testid="stChatMessage"] {{ background:#0c2040 !important; border:1px solid {C_GOLD}33 !important; border-radius:12px !important; color:white !important; }}
[data-testid="stChatInput"] textarea {{ background:#0c2040 !important; border:1px solid {C_GOLD} !important; color:white !important; border-radius:10px !important; font-size:15px !important; }}
 
/* TEXTO */
h1,h2,h3,h4 {{ color:white !important; }}
p,span,label {{ color:#cce0ff; }}
.stMarkdown p {{ color:#cce0ff; font-size:15px; }}
code {{ background:#0c2040 !important; color:{C_GOLD} !important; border-radius:6px !important; }}
 
/* SEPARADOR */
hr {{ border-color:{C_GOLD}44 !important; margin:20px 0 !important; }}
 
/* SCROLLBAR */
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-track {{ background:#071528; }}
::-webkit-scrollbar-thumb {{ background:{C_GOLD}88; border-radius:4px; }}
 
/* SELECTBOX */
[data-testid="stSelectbox"] > div > div {{
    background: #0c2040 !important;
    border: 1px solid {C_GOLD}66 !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}}
 
/* CAPTION */
.stCaption {{ color:#7aaad8 !important; font-size:13px !important; }}
</style>""", unsafe_allow_html=True)
 
# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <div style="display:flex;align-items:center;gap:28px">
    <div style="background:white;border-radius:12px;padding:10px 18px;flex-shrink:0">
      <img src="https://www.creditex.com.pe/wp-content/uploads/2021/03/logo-creditex.png"
           height="55"
           onerror="this.outerHTML='<div style=font-weight:800;color:{C_NAVY};font-size:20px;letter-spacing:2px>CREDITEX</div>'">
    </div>
    <div>
      <div class="header-title">🧵 Sistema Inteligente de Predicción y Recomendación Preventiva</div>
      <div class="header-sub">Tejido Plano &nbsp;·&nbsp; CREDITEX S.A.A. &nbsp;·&nbsp; Universidad Nacional de Ingeniería (UNI) &nbsp;·&nbsp; Datos reales de producción &nbsp;·&nbsp; CONCYTEC 2026</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
 
# ── CONEXIÓN ─────────────────────────────────────────────────────────────────
if "cloud_connected" not in st.session_state:
    st.session_state.update({"cloud_connected":False,"conn_info":None,"df_cloud":None,"summary":None,"opts":None})
 
with st.expander("☁️ Conexión a base de datos en la nube — CREDITEX", expanded=not st.session_state.cloud_connected):
    col_cfg, col_btn = st.columns([3,1])
    with col_cfg:
        st.markdown(f"""<div style="display:flex;gap:28px;flex-wrap:wrap;font-size:14px;color:#8ab0d8;margin:8px 0">
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
            with st.spinner(f"Descargando {conn['rows_available']:,} registros reales..."):
                df = fetch_records(n=2000)
            st.session_state.df_cloud = df
            with st.spinner("Calculando KPIs..."):
                st.session_state.summary = get_summary_stats()
                st.session_state.opts    = get_filter_options()
            st.success(f"✅ Conectado · {len(df):,} registros · {conn['latency_ms']} ms · {conn['rango_fechas']}")
            st.rerun()
 
    if st.session_state.cloud_connected and st.session_state.conn_info:
        c = st.session_state.conn_info
        st.markdown(f"""<div class="cloud-badge">
          🟢 <b>Conectado</b> &nbsp;|&nbsp; <b>{c['host']}</b> &nbsp;|&nbsp;
          Latencia: <b>{c['latency_ms']} ms</b> &nbsp;|&nbsp; Registros: <b>{c['rows_available']:,}</b> &nbsp;|&nbsp;
          Telares: <b>{c['telares_activos']}</b> &nbsp;|&nbsp; Período: <b>{c['rango_fechas']}</b>
        </div>""", unsafe_allow_html=True)
 
if not st.session_state.cloud_connected:
    st.warning("⚠️ Conecta primero a la base de datos en la nube para cargar los datos reales de CREDITEX.")
    st.stop()
 
df_raw = st.session_state.df_cloud
opts   = st.session_state.opts
 
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
    st.markdown(f"""<div style="text-align:center;padding:14px 0 18px;border-bottom:1px solid {C_GOLD}55;margin-bottom:18px">
        <div style="background:white;border-radius:10px;padding:8px 14px;display:inline-block">
          <img src="https://www.creditex.com.pe/wp-content/uploads/2021/03/logo-creditex.png"
               height="44"
               onerror="this.outerHTML='<b style=color:{C_NAVY};font-size:18px>CREDITEX</b>'">
        </div>
        <div style="font-size:12px;color:{C_GOLD};margin-top:8px;font-weight:600">Sistema Inteligente de Producción</div>
    </div>""", unsafe_allow_html=True)
 
    st.markdown("### ⚙️ Parámetros del telar")
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
    st.markdown(f"<div style='font-size:13px;color:{C_GOLD};font-weight:700'>☁️ {CLOUD_CONFIG['database']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:13px;color:#8ab0d8'>📊 {len(df_raw):,} registros</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:13px;color:#8ab0d8'>🤖 Precisión: <b style='color:{C_GOLD}'>{precision*100:.1f}%</b></div>", unsafe_allow_html=True)
 
# ── PREDICCIÓN ───────────────────────────────────────────────────────────────
ent = {
    "Temperatura":temperatura,"Humedad":humedad,"RPM_Telar":rpm,
    "Eficiencia_Telar":eficiencia,"Tension_Urdimbre":tension_urd,
    "Roturas_Urdimbre":roturas_urd,"Roturas_Trama":roturas_tram,
    "Metros_Defectuosos":metros_def,"Tiempo_Parada_Min":t_parada,
    "Tiempo_Reparacion_Min":t_reparac,
    "Dias_Desde_Mantenimiento_Preventivo":dias_mant,
    "Turno_A":int(turno=="A"),"Turno_B":int(turno=="B"),"Turno_C":int(turno=="C"),
}
prob = modelo.predict_proba(pd.DataFrame([ent])[columnas])[0][1]
riesgo_pct = prob * 100
if riesgo_pct >= 55:  nivel,color_g,emoji = "ALTO","#ff3333","🔴"
elif riesgo_pct >= 35: nivel,color_g,emoji = "MEDIO",C_GOLD,"🟡"
else:                  nivel,color_g,emoji = "BAJO","#00cc55","🟢"
 
PLT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,32,64,0.7)",
           font={"color":"#cce0ff","size":13},
           xaxis={"gridcolor":"#1a3f6f","color":"#8ab0d8"},
           yaxis={"gridcolor":"#1a3f6f","color":"#8ab0d8"})
 
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 Monitor en tiempo real","💡 Recomendaciones","📈 Análisis","☁️ Datos de la nube","🤖 Asistente IA"
])
 
# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🎯 Riesgo de defecto", f"{riesgo_pct:.1f}%", f"{emoji} Nivel {nivel}",
              delta_color="inverse" if nivel=="BAJO" else "normal")
    c2.metric("🌡 Temperatura", f"{temperatura:.1f}°C")
    c3.metric("⚙️ RPM Telar",   f"{rpm}")
    c4.metric("📊 Eficiencia",  f"{eficiencia:.1f}%")
 
    st.markdown("<hr>", unsafe_allow_html=True)
    cg,ca = st.columns(2)
 
    with cg:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=riesgo_pct,
            delta={"reference":35,"suffix":"%",
                   "increasing":{"color":"#ff3333"},"decreasing":{"color":"#00cc55"}},
            title={"text":f"<b style='font-size:18px'>Riesgo de Defecto</b><br>"
                          f"<span style='font-size:16px;color:{color_g}'>{emoji} Nivel {nivel}</span>",
                   "font":{"color":"white","size":18}},
            number={"suffix":"%","font":{"size":64,"color":"white"}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#8ab0d8",
                        "tickfont":{"color":"#8ab0d8","size":13}},
                "bar":{"color":color_g,"thickness":0.3},
                "bgcolor":C_DARK,
                "bordercolor":C_GOLD,"borderwidth":2,
                "steps":[{"range":[0,35],"color":"#031a0a"},
                         {"range":[35,55],"color":"#1a1000"},
                         {"range":[55,100],"color":"#1a0505"}],
                "threshold":{"line":{"color":C_GOLD,"width":4},"thickness":0.9,"value":riesgo_pct},
            }
        ))
        fig.update_layout(height=340, margin=dict(t=80,b=20,l=30,r=30),
                          paper_bgcolor="rgba(0,0,0,0)", font={"color":"white"})
        st.plotly_chart(fig, use_container_width=True)
 
    with ca:
        st.markdown(f"<h4 style='color:{C_GOLD}'>Estado del proceso</h4>", unsafe_allow_html=True)
        if nivel=="ALTO":
            st.markdown(f'<div class="alert-danger"><b>⚠️ ALERTA CRÍTICA</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b><br>Intervención inmediata del supervisor de turno.</div>', unsafe_allow_html=True)
        elif nivel=="MEDIO":
            st.markdown(f'<div class="alert-warning"><b>⚡ PRECAUCIÓN</b><br>Riesgo moderado: <b>{riesgo_pct:.1f}%</b><br>Aplicar ajustes preventivos.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><b>✅ OPERACIÓN NORMAL</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b><br>Parámetros dentro del rango.</div>', unsafe_allow_html=True)
 
        st.markdown(f"<h4 style='color:{C_GOLD};margin-top:20px'>Parámetros del telar</h4>", unsafe_allow_html=True)
        params_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:15px;color:#cce0ff">
          <tr style="border-bottom:1px solid {C_GOLD}44">
            <td style="padding:10px 14px;color:{C_GOLD};font-weight:700">Parámetro</td>
            <td style="padding:10px 14px;color:{C_GOLD};font-weight:700">Valor</td>
          </tr>
          <tr style="background:#0c2040"><td style="padding:10px 14px">🌡 Temperatura</td><td style="padding:10px 14px;font-weight:700">{temperatura}°C</td></tr>
          <tr style="background:#0d2244"><td style="padding:10px 14px">💧 Humedad</td><td style="padding:10px 14px;font-weight:700">{humedad}%</td></tr>
          <tr style="background:#0c2040"><td style="padding:10px 14px">⚙️ RPM</td><td style="padding:10px 14px;font-weight:700">{rpm}</td></tr>
          <tr style="background:#0d2244"><td style="padding:10px 14px">📊 Eficiencia</td><td style="padding:10px 14px;font-weight:700">{eficiencia}%</td></tr>
          <tr style="background:#0c2040"><td style="padding:10px 14px">🧵 Roturas urdimbre</td><td style="padding:10px 14px;font-weight:700">{roturas_urd}</td></tr>
          <tr style="background:#0d2244"><td style="padding:10px 14px">🧵 Roturas trama</td><td style="padding:10px 14px;font-weight:700">{roturas_tram}</td></tr>
        </table>"""
        st.markdown(params_html, unsafe_allow_html=True)
 
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{C_GOLD}'>Relación RPM vs Nivel de Riesgo — datos reales CREDITEX</h4>", unsafe_allow_html=True)
    fig2 = px.box(df_raw, x="Nivel_Riesgo", y="RPM_Telar", color="Nivel_Riesgo",
                  color_discrete_map={"Alto":"#ff3333","Medio":C_GOLD,"Bajo":"#00cc55"},
                  category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
    fig2.update_layout(height=300, showlegend=False, margin=dict(t=10,b=10), **PLT)
    st.plotly_chart(fig2, use_container_width=True)
 
# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<h3 style='color:{C_GOLD}'>💡 Recomendaciones preventivas del sistema experto</h3>", unsafe_allow_html=True)
    recs = []
    if roturas_urd > 8:
        recs.append(("🧵 Roturas de urdimbre elevadas", f"{roturas_urd} roturas. Verificar tensión y estado de hilos.", "alta"))
    if roturas_tram > 8:
        recs.append(("🧵 Roturas de trama elevadas", f"{roturas_tram} roturas. Revisar lanzadera.", "alta"))
    if eficiencia < 75:
        recs.append(("📊 Eficiencia baja", f"{eficiencia:.1f}%. Telares con eficiencia <75% tienen mayor tasa de rechazo.", "alta"))
    if t_parada > 120:
        recs.append(("⏸ Tiempo de parada excesivo", f"{t_parada} min. Investigar causa raíz.", "media"))
    if dias_mant > 60:
        recs.append(("🛠 Mantenimiento vencido", f"{dias_mant} días desde último mantenimiento. Programar urgente.", "alta"))
    if metros_def > 100:
        recs.append(("📏 Metros defectuosos elevados", f"{metros_def} m. Activar inspección detallada.", "alta"))
    if not recs:
        recs.append(("✅ Sistema en condiciones óptimas", "Parámetros dentro de rangos históricos normales de CREDITEX.", "baja"))
 
    tag_map   = {"alta":"tag-high","media":"tag-med","baja":"tag-low"}
    label_map = {"alta":"Prioridad Alta","media":"Prioridad Media","baja":"Prioridad Baja"}
    for titulo,texto,prio in recs:
        st.markdown(f"""<div class="rec-box">
            <b style="font-size:16px;color:white">{titulo}</b>
            <span class="{tag_map[prio]}" style="margin-left:12px">{label_map[prio]}</span><br>
            <span style="font-size:14px;margin-top:6px;display:block">{texto}</span>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{C_GOLD}'>Distribución de acciones recomendadas</h4>", unsafe_allow_html=True)
    acc_df = df_raw["Accion_Recomendada"].value_counts().reset_index()
    acc_df.columns = ["Acción","Frecuencia"]
    fig_acc = px.bar(acc_df, x="Frecuencia", y="Acción", orientation="h",
                     color="Frecuencia", color_continuous_scale=[[0,C_BLUE],[1,C_GOLD]])
    fig_acc.update_layout(height=240, coloraxis_showscale=False, margin=dict(t=10,b=10), **PLT)
    st.plotly_chart(fig_acc, use_container_width=True)
 
# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<h3 style='color:{C_GOLD}'>📈 Análisis de datos reales — CREDITEX</h3>", unsafe_allow_html=True)
    summ = st.session_state.summary
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total registros en BD", f"{summ['total_registros']:,}")
    k2.metric("Nivel alto de riesgo",  f"{summ['pct_nivel_alto']}%")
    k3.metric("Tasa rechazado",        f"{summ['pct_rechazado']}%")
    k4.metric("Precisión del modelo",  f"{precision*100:.1f}%")
    st.markdown("<hr>", unsafe_allow_html=True)
 
    ca2,cb2 = st.columns(2)
    with ca2:
        st.markdown(f"<h4 style='color:{C_GOLD}'>Variables más influyentes</h4>", unsafe_allow_html=True)
        top = importancias.head(8).reset_index()
        top.columns = ["Variable","Importancia"]
        fig_i = px.bar(top, x="Importancia", y="Variable", orientation="h",
                       color="Importancia", color_continuous_scale=[[0,C_BLUE],[1,C_GOLD]])
        fig_i.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=10,b=10), **PLT)
        st.plotly_chart(fig_i, use_container_width=True)
 
    with cb2:
        st.markdown(f"<h4 style='color:{C_GOLD}'>Nivel de riesgo por turno</h4>", unsafe_allow_html=True)
        tdf = df_raw.groupby(["Turno","Nivel_Riesgo"]).size().reset_index(name="n")
        fig_t = px.bar(tdf, x="Turno", y="n", color="Nivel_Riesgo",
                       color_discrete_map={"Alto":"#ff3333","Medio":C_GOLD,"Bajo":"#00cc55"},
                       barmode="stack")
        fig_t.update_layout(height=340, margin=dict(t=10,b=10), **PLT)
        st.plotly_chart(fig_t, use_container_width=True)
 
    cc2,cd2 = st.columns(2)
    with cc2:
        st.markdown(f"<h4 style='color:{C_GOLD}'>Tipo de defecto más frecuente</h4>", unsafe_allow_html=True)
        def_df = df_raw["Tipo_Defecto"].value_counts().reset_index()
        def_df.columns = ["Tipo","n"]
        fig_d = px.pie(def_df, values="n", names="Tipo",
                       color_discrete_sequence=[C_GOLD,C_BLUE,"#00cc55","#ff3333","#aa66ff","#00cccc","#ff6644"])
        fig_d.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                            font={"color":"#cce0ff","size":13}, margin=dict(t=10,b=10))
        st.plotly_chart(fig_d, use_container_width=True)
 
    with cd2:
        st.markdown(f"<h4 style='color:{C_GOLD}'>Eficiencia vs nivel de riesgo</h4>", unsafe_allow_html=True)
        fig_e = px.violin(df_raw, x="Nivel_Riesgo", y="Eficiencia_Telar", color="Nivel_Riesgo",
                          color_discrete_map={"Alto":"#ff3333","Medio":C_GOLD,"Bajo":"#00cc55"},
                          category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
        fig_e.update_layout(height=320, showlegend=False, margin=dict(t=10,b=10), **PLT)
        st.plotly_chart(fig_e, use_container_width=True)
 
    st.markdown(f"<h4 style='color:{C_GOLD}'>Resultado de inspección por cliente</h4>", unsafe_allow_html=True)
    cli_df = df_raw.groupby(["Cliente","Resultado_Inspeccion"]).size().reset_index(name="n")
    fig_cli = px.bar(cli_df, x="Cliente", y="n", color="Resultado_Inspeccion",
                     color_discrete_map={"Rechazado":"#ff3333","Observado":C_GOLD,"Aprobado":"#00cc55"},
                     barmode="stack")
    fig_cli.update_layout(height=300, margin=dict(t=10,b=10), **PLT)
    st.plotly_chart(fig_cli, use_container_width=True)
 
# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"<h3 style='color:{C_GOLD}'>☁️ Datos reales de producción — CREDITEX</h3>", unsafe_allow_html=True)
    st.code("SELECT * FROM dbo.registros_operativos ORDER BY Fecha DESC LIMIT 2000;", language="sql")
 
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
 
    col_d1,col_d2 = st.columns(2)
    csv = dfv.to_csv(index=False).encode("utf-8")
    col_d1.download_button("⬇️ Descargar CSV", csv, "creditex_produccion.csv", "text/csv")
    if col_d2.button("🔄 Re-sincronizar"):
        with st.spinner("Consultando Azure SQL..."):
            st.session_state.df_cloud = fetch_records(n=2000)
        st.success("✅ Datos actualizados"); st.rerun()
 
# ══ TAB 5 ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"<h3 style='color:{C_GOLD}'>🤖 Asistente de Producción CREDITEX</h3>", unsafe_allow_html=True)
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
        if any(x in pl for x in ["riesgo","defecto","prediccion"]):
            return f"Riesgo actual del telar **{telar_sel}**: **{riesgo_pct:.1f}%** (nivel **{nivel}**). Precisión del modelo: **{precision*100:.1f}%** sobre {len(df_raw):,} registros reales."
        if any(x in pl for x in ["nube","base","azure","sql","datos","registros"]):
            c = st.session_state.conn_info
            return f"- **Proveedor:** {CLOUD_CONFIG['provider']}\n- **Registros:** {c['rows_available']:,}\n- **Período:** {c['rango_fechas']}\n- **Telares activos:** {c['telares_activos']}\n- **Latencia:** {c['latency_ms']} ms"
        if any(x in pl for x in ["recomienda","acción","accion","hacer"]):
            if not recs: return "✅ El proceso opera correctamente según el histórico de CREDITEX."
            r = "**Recomendaciones prioritarias:**\n\n"
            for tit,txt,prio in recs[:3]: r += f"- **{tit}** ({prio}): {txt}\n\n"
            return r
        if any(x in pl for x in ["cliente","reclamo"]):
            return f"- Más reclamos: **{summ['cliente_mas_reclamos']}**\n- Tasa de rechazo: **{summ['pct_rechazado']}%**\n- Nivel alto: **{summ['pct_nivel_alto']}%**"
        if any(x in pl for x in ["modelo","precision"]):
            return f"**Random Forest** (150 árboles) · {len(df_raw):,} registros reales · Precisión: **{precision*100:.1f}%**"
        if any(x in pl for x in ["hola","buenas"]):
            return f"¡Hola! 😊 {len(df_raw):,} registros listos. Riesgo: **{riesgo_pct:.1f}%**."
        return f"Puedo ayudarte con: riesgo, nube, recomendaciones, clientes o modelo.\n\n**Resumen:** {riesgo_pct:.1f}% · {len(df_raw):,} registros · {precision*100:.1f}%"
 
    if pregunta := st.chat_input("Escribe tu consulta..."):
        st.session_state.chat_history.append({"role":"user","content":pregunta})
        with st.chat_message("user"): st.markdown(pregunta)
        resp = responder(pregunta)
        st.session_state.chat_history.append({"role":"assistant","content":resp})
        with st.chat_message("assistant"): st.markdown(resp)
 
    st.markdown(f"<div style='margin-top:16px;font-size:14px;color:{C_GOLD};font-weight:700'>Preguntas rápidas:</div>", unsafe_allow_html=True)
    cols_q = st.columns(4)
    pqs = ["¿Cuál es el riesgo actual?","¿Qué hay en la base de datos?","¿Qué se recomienda?","¿Cómo funciona el modelo?"]
    for i,pq in enumerate(pqs):
        if cols_q[i].button(pq, use_container_width=True):
            st.session_state.chat_history.append({"role":"user","content":pq})
            st.session_state.chat_history.append({"role":"assistant","content":responder(pq)})
            st.rerun()
 
# FOOTER
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""<div style='text-align:center;padding:12px;font-size:13px;color:{C_GOLD}88'>
    🧵 CREDITEX S.A.A. &nbsp;·&nbsp; Universidad Nacional de Ingeniería (UNI) &nbsp;·&nbsp;
    ☁️ Azure SQL Database &nbsp;·&nbsp; CONCYTEC 2026 &nbsp;·&nbsp; Prototipo v3.1
</div>""", unsafe_allow_html=True)
 
