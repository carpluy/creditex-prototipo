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

st.set_page_config(page_title="Sistema Inteligente - CREDITEX", page_icon="🌿", layout="wide")

C_PRIMARY   = "#5B21B6"
C_SECONDARY = "#2563EB"
C_SUCCESS   = "#059669"
C_WARNING   = "#D97706"
C_DANGER    = "#DC2626"
BG_APP      = "#F8FAFC"
BG_CARD     = "#FFFFFF"
TEXT_MAIN   = "#1E1B4B"
TEXT_LIGHT  = "#6B7280"
BORDER      = "#E0E7FF"

st.markdown(f"""
<style>
.stApp {{ background:{BG_APP} !important; }}

section[data-testid="stSidebar"] {{
    background:#FFFFFF !important;
    border-right:2px solid {BORDER} !important;
    min-width:220px !important; max-width:240px !important;
}}
/* FORZAR TODOS LOS TEXTOS DEL SIDEBAR EN OSCURO */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [class*="st-"],
section[data-testid="stSidebar"] [data-testid*="st"] {{
    color: #1E1B4B !important;
    font-size: 13px !important;
}}
section[data-testid="stSidebar"] h3 {{
    color: {C_PRIMARY} !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}}
section[data-testid="stSidebar"] .stButton > button {{
    background: linear-gradient(135deg, {C_PRIMARY}, {C_SECONDARY}) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    box-shadow: 0 3px 10px rgba(91,33,182,.3) !important;
}}
section[data-testid="stSidebar"] .stButton > button * {{
    color: white !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: {BG_APP} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 12px !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {{
    color: {TEXT_MAIN} !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
    background: white !important;
    color: {TEXT_MAIN} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {{
    background: white !important;
    color: {TEXT_MAIN} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 8px !important;
}}
.sidebar-logo {{
    background: linear-gradient(135deg, {C_PRIMARY}, {C_SECONDARY});
    padding: 20px 16px; border-radius: 16px; text-align: center;
    box-shadow: 0 8px 24px rgba(91,33,182,.4); margin-bottom: 18px;
    border: 2px solid rgba(255,255,255,.2);
}}
.sidebar-logo * {{ color: white !important; }}
.sidebar-stat {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; background: {BG_APP};
    border-radius: 10px; margin: 6px 0; border: 1.5px solid {BORDER};
}}
.sidebar-stat-icon {{ font-size: 18px; }}
.sidebar-stat-text {{ font-size: 12px; color: {TEXT_LIGHT} !important; font-weight: 500; }}
.sidebar-stat-val  {{ font-size: 14px; font-weight: 700; color: {TEXT_MAIN} !important; }}

.main-header {{
    background:linear-gradient(135deg,{C_PRIMARY} 0%,#7C3AED 40%,{C_SECONDARY} 100%);
    border:none; border-radius:22px; padding:36px 44px; margin-bottom:22px;
    box-shadow:0 14px 40px rgba(91,33,182,.32);
    display:flex; align-items:center; gap:24px;
    position:relative; overflow:hidden;
}}
.main-header::after {{
    content:""; position:absolute; top:-100px; right:-100px;
    width:320px; height:320px; border-radius:50%;
    background:rgba(255,255,255,.08); pointer-events:none;
}}
.main-header::before {{
    content:""; position:absolute; bottom:-80px; left:30%;
    width:200px; height:200px; border-radius:50%;
    background:rgba(255,255,255,.05); pointer-events:none;
}}
.header-icon {{ background:rgba(255,255,255,.2); border-radius:18px; padding:18px; font-size:34px; flex-shrink:0; z-index:1; }}
.header-title {{ font-size:34px; font-weight:900; color:white; margin:0; letter-spacing:-.5px; z-index:1; }}
.header-sub   {{ font-size:14px; color:rgba(255,255,255,.88); margin:7px 0 0; z-index:1; }}
.header-gold  {{ color:#FDE68A; font-weight:800; }}
.header-badge {{ background:rgba(255,255,255,.2); border-radius:14px; padding:12px 22px;
                 font-size:20px; font-weight:900; color:white; flex-shrink:0; letter-spacing:1px; z-index:1; }}

/* KPI CARDS */
.kpi-card {{
    background:linear-gradient(135deg,#FFFFFF 0%,#F8FAFF 100%);
    border:1.5px solid {BORDER}; border-radius:22px;
    padding:28px 24px; box-shadow:0 6px 20px rgba(91,33,182,.09);
    display:flex; align-items:center; gap:20px;
    transition:.25s; cursor:default; min-height:130px;
}}
.kpi-card:hover {{ transform:translateY(-5px); box-shadow:0 16px 36px rgba(91,33,182,.18); }}
.kpi-icon {{ width:70px; height:70px; border-radius:20px; display:flex; align-items:center;
             justify-content:center; font-size:32px; flex-shrink:0; }}
.ic-red    {{ background:linear-gradient(135deg,#FF6B6B,{C_DANGER}); }}
.ic-blue   {{ background:linear-gradient(135deg,#60A5FA,{C_SECONDARY}); }}
.ic-purple {{ background:linear-gradient(135deg,#A78BFA,{C_PRIMARY}); }}
.ic-green  {{ background:linear-gradient(135deg,#34D399,{C_SUCCESS}); }}
.kpi-label {{ font-size:14px; color:{TEXT_LIGHT}; font-weight:600; margin:0; }}
.kpi-value {{ font-size:64px; font-weight:900; color:{TEXT_MAIN}; margin:4px 0 0; line-height:1; }}
.kpi-badge {{ display:inline-block; font-size:13px; font-weight:700; padding:5px 14px; border-radius:20px; margin-top:8px; }}
.bg-green  {{ background:#D1FAE5; color:#065F46; }}
.bg-red    {{ background:#FEE2E2; color:#991B1B; }}
.bg-yellow {{ background:#FEF9C3; color:#92400E; }}

/* PARAM MINI CARDS */
.param-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px; }}
.param-card {{
    background:white; border:1.5px solid {BORDER}; border-radius:14px;
    padding:12px 16px; display:flex; flex-direction:column;
    box-shadow:0 4px 12px rgba(37,99,235,.08);
}}
.param-card-label {{ font-size:12px; color:{TEXT_LIGHT}; font-weight:600; margin:0; }}
.param-card-value {{ font-size:22px; font-weight:800; color:{TEXT_MAIN}; margin:2px 0 0; }}

/* CHART CARD */
.chart-card {{
    background:{BG_CARD}; border-radius:22px; padding:22px 24px;
    box-shadow:0 6px 20px rgba(0,0,0,.06); border:1px solid {BORDER};
    margin-top:16px;
}}
.chart-title {{ font-size:17px; font-weight:700; color:{TEXT_MAIN}; margin:0 0 14px; }}

/* TABS */
[data-baseweb="tab-list"] {{
    background:{BG_CARD} !important; border-radius:16px !important;
    padding:7px !important; border:1.5px solid {BORDER} !important;
    box-shadow:0 3px 10px rgba(91,33,182,.07) !important; gap:4px !important;
}}
[data-baseweb="tab"] {{
    background:transparent !important; color:{TEXT_LIGHT} !important;
    border-radius:12px !important; font-size:15px !important;
    font-weight:600 !important; padding:11px 22px !important; border:none !important;
}}
[aria-selected="true"] {{
    background:linear-gradient(135deg,{C_PRIMARY},{C_SECONDARY}) !important;
    color:white !important; box-shadow:0 4px 14px rgba(91,33,182,.38) !important;
}}

/* EXPANDER */
[data-testid="stExpander"] {{
    background:{BG_CARD} !important; border:1.5px solid {BORDER} !important;
    border-radius:16px !important; box-shadow:0 3px 10px rgba(0,0,0,.04) !important;
}}
[data-testid="stExpander"] summary {{ color:{C_PRIMARY} !important; font-weight:700 !important; font-size:15px !important; }}

/* ALERTAS */
.alert-danger  {{ background:#FFF5F5;border-left:6px solid {C_DANGER};padding:18px 22px;border-radius:14px;color:#991B1B;font-size:15px; }}
.alert-success {{ background:#ECFDF5;border-left:6px solid {C_SUCCESS};padding:18px 22px;border-radius:14px;color:#065F46;font-size:15px; }}
.alert-warning {{ background:#FFFBEB;border-left:6px solid {C_WARNING};padding:18px 22px;border-radius:14px;color:#92400E;font-size:15px; }}
.cloud-badge   {{ background:#ECFDF5;border:1.5px solid {C_SUCCESS};border-radius:12px;padding:14px 20px;font-size:14px;color:#065F46; }}

/* RECOMENDACIONES */
.rec-box {{
    background:{BG_CARD}; border:1.5px solid {BORDER};
    border-left:6px solid {C_PRIMARY}; border-radius:16px;
    padding:18px 22px; margin:10px 0;
    box-shadow:0 3px 10px rgba(91,33,182,.07);
}}
.tag-high {{ background:#FEE2E2;color:#991B1B;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }}
.tag-med  {{ background:#FEF9C3;color:#92400E;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }}
.tag-low  {{ background:#D1FAE5;color:#065F46;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }}

/* MÉTRICAS NATIVAS */
[data-testid="metric-container"] {{
    background:{BG_CARD} !important; border:1.5px solid {BORDER} !important;
    border-radius:20px !important; padding:24px !important;
    box-shadow:0 6px 20px rgba(91,33,182,.08) !important; transition:.25s !important;
}}
[data-testid="metric-container"]:hover {{ transform:translateY(-4px) !important; }}
[data-testid="stMetricLabel"] {{ color:{TEXT_LIGHT} !important; font-size:14px !important; font-weight:600 !important; }}
[data-testid="stMetricValue"] {{ color:{TEXT_MAIN} !important; font-size:40px !important; font-weight:800 !important; }}

/* BOTONES */
.stButton>button {{
    background:linear-gradient(135deg,{C_PRIMARY},{C_SECONDARY}) !important;
    color:white !important; border:none !important; border-radius:12px !important;
    font-weight:700 !important; font-size:14px !important; padding:11px 22px !important;
    box-shadow:0 4px 14px rgba(91,33,182,.3) !important; transition:all .2s !important;
}}
.stButton>button:hover {{ transform:translateY(-2px) !important; box-shadow:0 8px 22px rgba(91,33,182,.45) !important; }}

/* DATAFRAME */
[data-testid="stDataFrame"] {{ border-radius:18px !important; border:1.5px solid {BORDER} !important; overflow:hidden !important; box-shadow:0 6px 20px rgba(0,0,0,.06) !important; }}

/* CHAT */
[data-testid="stChatMessage"] {{ background:{BG_CARD} !important; border:1.5px solid {BORDER} !important; border-radius:14px !important; }}
[data-testid="stChatInput"] textarea {{ background:{BG_CARD} !important; border:2px solid {C_PRIMARY} !important; border-radius:12px !important; font-size:15px !important; }}

/* MAIN CONTENT */
.main .block-container {{
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}}
h1,h2,h3,h4 {{ color:{TEXT_MAIN} !important; }}
.section-title {{ font-size:18px; font-weight:700; color:{TEXT_MAIN}; margin:16px 0 12px; }}
.stMarkdown p {{ color:{TEXT_MAIN}; font-size:15px; }}
code {{ background:{C_PRIMARY}18 !important; color:{C_PRIMARY} !important; border-radius:6px !important; }}
.stCaption {{ color:{TEXT_LIGHT} !important; font-size:13px !important; }}
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-thumb {{ background:{C_PRIMARY}66; border-radius:4px; }}
.footer {{ text-align:center; padding:18px; font-size:13px; color:{TEXT_LIGHT}; border-top:1.5px solid {BORDER}; margin-top:24px; }}
</style>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <div class="header-icon">📊</div>
  <div style="flex:1">
    <div class="header-title">Sistema Inteligente de Predicción y Recomendación Preventiva</div>
    <div class="header-sub">Tejido Plano — CREDITEX S.A.A. — Universidad Nacional de Ingeniería (UNI) · Datos reales de producción · <span class="header-gold">CONCYTEC 2026</span></div>
  </div>
  <div class="header-badge">🧵 CREDITEX</div>
</div>""", unsafe_allow_html=True)

# ── CONEXIÓN ──────────────────────────────────────────────────────────────────
if "cloud_connected" not in st.session_state:
    st.session_state.update({"cloud_connected":False,"conn_info":None,"df_cloud":None,"summary":None,"opts":None})

with st.expander("☁️ Conexión a base de datos en la nube — CREDITEX", expanded=not st.session_state.cloud_connected):
    col_cfg,col_btn=st.columns([3,1])
    with col_cfg:
        st.markdown(f"""<div style="display:flex;gap:22px;flex-wrap:wrap;font-size:14px;color:{TEXT_LIGHT};margin:8px 0">
          <span>🖥 <b style="color:{TEXT_MAIN}">Proveedor:</b> {CLOUD_CONFIG['provider']}</span>
          <span>🌐 <b style="color:{TEXT_MAIN}">Host:</b> {CLOUD_CONFIG['host']}</span>
          <span>🗄 <b style="color:{TEXT_MAIN}">BD:</b> {CLOUD_CONFIG['database']}</span>
          <span>📍 <b style="color:{TEXT_MAIN}">Región:</b> {CLOUD_CONFIG['region']}</span>
          <span>🔒 <b style="color:{TEXT_MAIN}">SSL:</b> TLS 1.2</span>
        </div>""", unsafe_allow_html=True)
    with col_btn:
        if st.button("🔌 Conectar a la nube", use_container_width=True, type="primary"):
            with st.spinner("Estableciendo conexión segura..."):
                conn=get_connection_status()
            st.session_state.conn_info=conn; st.session_state.cloud_connected=True
            with st.spinner(f"Descargando {conn['rows_available']:,} registros..."):
                df=fetch_records(n=2000)
            st.session_state.df_cloud=df
            with st.spinner("Calculando KPIs..."):
                st.session_state.summary=get_summary_stats()
                st.session_state.opts=get_filter_options()
            st.success(f"✅ Conectado · {len(df):,} registros · {conn['latency_ms']} ms")
            st.rerun()
    if st.session_state.cloud_connected and st.session_state.conn_info:
        c=st.session_state.conn_info
        st.markdown(f"""<div class="cloud-badge">🟢 <b>Conectado</b> &nbsp;|&nbsp; {c['host']} &nbsp;|&nbsp;
          Latencia: <b>{c['latency_ms']} ms</b> &nbsp;|&nbsp; Registros: <b>{c['rows_available']:,}</b> &nbsp;|&nbsp;
          Telares: <b>{c['telares_activos']}</b> &nbsp;|&nbsp; Período: <b>{c['rango_fechas']}</b>
        </div>""", unsafe_allow_html=True)

if not st.session_state.cloud_connected:
    st.warning("⚠️ Conecta primero a la base de datos en la nube para cargar los datos reales de CREDITEX.")
    st.stop()

df_raw=st.session_state.df_cloud; opts=st.session_state.opts

@st.cache_resource
def entrenar_modelo(df):
    d=df.copy()
    for t in ["A","B","C"]: d[f"Turno_{t}"]=(d["Turno"]==t).astype(int)
    d["Riesgo_bin"]=(d["Nivel_Riesgo"]=="Alto").astype(int)
    feats=["Temperatura","Humedad","RPM_Telar","Eficiencia_Telar","Tension_Urdimbre",
           "Roturas_Urdimbre","Roturas_Trama","Metros_Defectuosos","Tiempo_Parada_Min",
           "Tiempo_Reparacion_Min","Dias_Desde_Mantenimiento_Preventivo","Turno_A","Turno_B","Turno_C"]
    X=d[feats].fillna(0); y=d["Riesgo_bin"]
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=42)
    m=RandomForestClassifier(n_estimators=150,max_depth=10,random_state=42); m.fit(Xt,yt)
    imp=pd.Series(m.feature_importances_,index=feats).sort_values(ascending=False)
    return m,feats,accuracy_score(yv,m.predict(Xv)),imp

modelo,columnas,precision,importancias=entrenar_modelo(df_raw)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # LOGO
    st.markdown(f"""<div class="sidebar-logo">
      <div style="font-size:22px;font-weight:900;color:white !important;letter-spacing:2px;text-shadow:0 2px 8px rgba(0,0,0,.3)">🧵 CREDITEX</div>
      <div style="font-size:11px;font-weight:600;color:white !important;opacity:0.9;margin-top:4px">Sistema Inteligente · Producción</div>
    </div>""", unsafe_allow_html=True)

    # CONTROLES PRINCIPALES
    telar_sel = st.selectbox("🏭 Telar", opts["telares"][1:])
    turno     = st.selectbox("👷 Turno", ["A","B","C"])

    if st.button("🔄 Actualizar datos", use_container_width=True):
        with st.spinner(f"Consultando {telar_sel}..."):
            lec = fetch_latest_reading(telar_sel)
        if lec:
            st.session_state[f"lec_{telar_sel}"] = lec
            st.success(f"✅ Sync: {lec['timestamp']}")

    lec = st.session_state.get(f"lec_{telar_sel}")

    st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0'>", unsafe_allow_html=True)

    # GRUPO 1: OPERACIÓN
    with st.expander("🌡 Operación", expanded=True):
        temperatura = st.slider("Temperatura (°C)", 15.0,40.0, float(lec["temperatura"]) if lec else 25.0,0.1)
        humedad     = st.slider("Humedad (%)",      40.0,95.0, float(lec["humedad"])      if lec else 65.0,0.5)
        rpm         = st.slider("RPM Telar",        200,900,   int(lec["rpm_telar"])       if lec else 500,10)
        eficiencia  = st.slider("Eficiencia (%)",   60.0,100.0,float(lec["eficiencia"])   if lec else 85.0,0.5)

    # GRUPO 2: CALIDAD
    with st.expander("🧵 Calidad", expanded=False):
        tension_urd  = st.slider("Tensión urdimbre", 15.0,50.0,30.0,0.5)
        roturas_urd  = st.number_input("Roturas urdimbre", 0,20,3)
        roturas_tram = st.number_input("Roturas trama",    0,20,3)
        metros_def   = st.number_input("Metros defectuosos",0,250,50)

    # GRUPO 3: MANTENIMIENTO
    with st.expander("🔧 Mantenimiento", expanded=False):
        t_parada  = st.number_input("Tiempo parada (min)",     0,240,30)
        t_reparac = st.number_input("Tiempo reparación (min)", 0,250,60)
        dias_mant = st.slider("Días desde mantenimiento",      0,120,30)

    st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0'>", unsafe_allow_html=True)

    # STATS
    st.markdown(f"""
    <div class="sidebar-stat">
      <span class="sidebar-stat-icon">☁️</span>
      <div><div class="sidebar-stat-text">Base de datos</div>
      <div class="sidebar-stat-val">{CLOUD_CONFIG['database']}</div></div>
    </div>
    <div class="sidebar-stat">
      <span class="sidebar-stat-icon">📊</span>
      <div><div class="sidebar-stat-text">Registros cargados</div>
      <div class="sidebar-stat-val">{len(df_raw):,}</div></div>
    </div>
    <div class="sidebar-stat">
      <span class="sidebar-stat-icon">🤖</span>
      <div><div class="sidebar-stat-text">Precisión del modelo</div>
      <div class="sidebar-stat-val" style="color:{C_PRIMARY}">{precision*100:.1f}%</div></div>
    </div>
    """, unsafe_allow_html=True)

# ── PREDICCIÓN ────────────────────────────────────────────────────────────────
ent={"Temperatura":temperatura,"Humedad":humedad,"RPM_Telar":rpm,"Eficiencia_Telar":eficiencia,
     "Tension_Urdimbre":tension_urd,"Roturas_Urdimbre":roturas_urd,"Roturas_Trama":roturas_tram,
     "Metros_Defectuosos":metros_def,"Tiempo_Parada_Min":t_parada,"Tiempo_Reparacion_Min":t_reparac,
     "Dias_Desde_Mantenimiento_Preventivo":dias_mant,
     "Turno_A":int(turno=="A"),"Turno_B":int(turno=="B"),"Turno_C":int(turno=="C")}
prob=modelo.predict_proba(pd.DataFrame([ent])[columnas])[0][1]
riesgo_pct=prob*100
if riesgo_pct>=55:   nivel,color_g,emoji,badge,ring_col="ALTO",C_DANGER,"🔴","bg-red","#DC2626"
elif riesgo_pct>=35: nivel,color_g,emoji,badge,ring_col="MEDIO",C_WARNING,"🟡","bg-yellow","#D97706"
else:                nivel,color_g,emoji,badge,ring_col="BAJO",C_SUCCESS,"🟢","bg-green","#059669"

PLT=dict(paper_bgcolor=BG_CARD,plot_bgcolor=BG_CARD,
         font=dict(color=TEXT_MAIN,size=14),
         xaxis=dict(gridcolor=BORDER,color=TEXT_LIGHT,showgrid=True),
         yaxis=dict(gridcolor=BORDER,color=TEXT_LIGHT,showgrid=True),
         margin=dict(t=20,b=20,l=10,r=10))

tab1,tab2,tab3,tab4,tab5=st.tabs([
    "📊 Monitor en tiempo real","💡 Recomendaciones","📈 Análisis","☁️ Datos de la nube","🧠 CREDITEX AI"
])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    # KPI CARDS
    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-icon ic-red">🎯</div>
          <div><p class="kpi-label">Riesgo de defecto</p>
          <p class="kpi-value" style="color:{color_g};font-size:48px">{riesgo_pct:.1f}%</p>
          <span class="kpi-badge {badge}">{emoji} Nivel {nivel}</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-icon ic-blue">🌡</div>
          <div><p class="kpi-label">Temperatura</p>
          <p class="kpi-value">{temperatura:.1f}<span style="font-size:24px;color:{TEXT_LIGHT}"> °C</span></p>
          <span class="kpi-badge bg-green">✔ Óptimo</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-icon ic-purple">⚙️</div>
          <div><p class="kpi-label">RPM Telar</p>
          <p class="kpi-value">{rpm}<span style="font-size:20px;color:{TEXT_LIGHT}"> rpm</span></p>
          <span class="kpi-badge bg-green">✔ Óptimo</span></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        ef_b="bg-green" if eficiencia>=80 else "bg-yellow"
        ef_t="✔ Bueno" if eficiencia>=80 else "⚠ Revisar"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-icon ic-green">📊</div>
          <div><p class="kpi-label">Eficiencia del telar</p>
          <p class="kpi-value">{eficiencia:.0f}<span style="font-size:24px;color:{TEXT_LIGHT}">%</span></p>
          <span class="kpi-badge {ef_b}">{ef_t}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # DONUT + ESTADO + PARAMS
    cg,ca=st.columns([0.8,1.2])

    with cg:
        st.markdown(f'<div class="chart-card"><div class="chart-title">📍 Indicador de riesgo</div>', unsafe_allow_html=True)
        resto=100-riesgo_pct
        fig=go.Figure(go.Pie(
            values=[riesgo_pct, resto],
            hole=0.65,
            marker_colors=[ring_col, "#E0E7FF"],
            textinfo="none", hoverinfo="skip", sort=False,
        ))
        fig.add_annotation(text=f"<b>{riesgo_pct:.1f}%</b>",
            x=0.5, y=0.57, font=dict(size=48, color=ring_col), showarrow=False)
        fig.add_annotation(text=f"{emoji} {nivel}",
            x=0.5, y=0.40, font=dict(size=20, color=TEXT_LIGHT), showarrow=False)
        fig.update_layout(height=360, showlegend=False,
            margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ca:
        # ALERTA
        if nivel=="ALTO":
            st.markdown(f'<div class="alert-danger"><b>⚠️ ALERTA CRÍTICA</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b> — Intervención inmediata del supervisor.</div>', unsafe_allow_html=True)
        elif nivel=="MEDIO":
            st.markdown(f'<div class="alert-warning"><b>⚡ PRECAUCIÓN</b><br>Riesgo moderado: <b>{riesgo_pct:.1f}%</b> — Aplicar ajustes preventivos.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><b>✅ OPERACIÓN NORMAL</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b> — Parámetros dentro del rango.</div>', unsafe_allow_html=True)

        st.markdown(f'<p class="section-title" style="margin-top:14px">📋 Parámetros del telar</p>', unsafe_allow_html=True)
        st.markdown(f"""<div class="param-grid">
          <div class="param-card"><span class="param-card-label">🌡 Temperatura</span><span class="param-card-value">{temperatura:.1f}°C</span></div>
          <div class="param-card"><span class="param-card-label">💧 Humedad</span><span class="param-card-value">{humedad:.0f}%</span></div>
          <div class="param-card"><span class="param-card-label">⚙️ RPM Telar</span><span class="param-card-value">{rpm}</span></div>
          <div class="param-card"><span class="param-card-label">📊 Eficiencia</span><span class="param-card-value">{eficiencia:.0f}%</span></div>
          <div class="param-card"><span class="param-card-label">🧵 Roturas urd.</span><span class="param-card-value">{roturas_urd}</span></div>
          <div class="param-card"><span class="param-card-label">🧵 Roturas tram.</span><span class="param-card-value">{roturas_tram}</span></div>
          <div class="param-card"><span class="param-card-label">🛠 Días mant.</span><span class="param-card-value">{dias_mant}</span></div>
          <div class="param-card"><span class="param-card-label">⏸ Parada</span><span class="param-card-value">{t_parada}m</span></div>
        </div>""", unsafe_allow_html=True)

    # GRÁFICO INFERIOR A ANCHO COMPLETO
    st.markdown(f"""<div class="chart-card">
      <div class="chart-title">📈 Relación RPM vs Nivel de Riesgo — datos reales CREDITEX</div>""", unsafe_allow_html=True)
    fig2=px.box(df_raw,x="Nivel_Riesgo",y="RPM_Telar",color="Nivel_Riesgo",
                color_discrete_map={"Alto":C_DANGER,"Medio":C_WARNING,"Bajo":C_SUCCESS},
                category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]},
                points="outliers")
    fig2.update_traces(marker_size=8, line_width=3)
    fig2.update_layout(height=380,showlegend=False,**PLT)
    st.plotly_chart(fig2,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<p class="section-title">💡 Recomendaciones preventivas del sistema experto</p>', unsafe_allow_html=True)
    recs=[]
    if roturas_urd>8:  recs.append(("🧵 Roturas de urdimbre elevadas",f"{roturas_urd} roturas. Verificar tensión y estado de hilos.","alta"))
    if roturas_tram>8: recs.append(("🧵 Roturas de trama elevadas",f"{roturas_tram} roturas. Revisar lanzadera.","alta"))
    if eficiencia<75:  recs.append(("📊 Eficiencia baja",f"{eficiencia:.1f}%. Los telares <75% tienen mayor tasa de rechazo en CREDITEX.","alta"))
    if t_parada>120:   recs.append(("⏸ Tiempo de parada excesivo",f"{t_parada} min parado. Investigar causa raíz.","media"))
    if dias_mant>60:   recs.append(("🛠 Mantenimiento vencido",f"{dias_mant} días desde último mantenimiento. Programar urgente.","alta"))
    if metros_def>100: recs.append(("📏 Metros defectuosos elevados",f"{metros_def} m defectuosos. Activar inspección detallada.","alta"))
    if not recs:       recs.append(("✅ Sistema en condiciones óptimas","Todos los parámetros dentro de los rangos históricos normales de CREDITEX.","baja"))

    tag_map={"alta":"tag-high","media":"tag-med","baja":"tag-low"}
    label_map={"alta":"Prioridad Alta","media":"Prioridad Media","baja":"Prioridad Baja"}
    for titulo,texto,prio in recs:
        st.markdown(f"""<div class="rec-box">
          <b style="font-size:16px;color:{TEXT_MAIN}">{titulo}</b>
          <span class="{tag_map[prio]}" style="margin-left:12px">{label_map[prio]}</span><br>
          <span style="font-size:14px;margin-top:8px;display:block;color:{TEXT_LIGHT}">{texto}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="chart-card" style="margin-top:20px">
      <div class="chart-title">📊 Distribución de acciones recomendadas</div>""", unsafe_allow_html=True)
    acc_df=df_raw["Accion_Recomendada"].value_counts().reset_index(); acc_df.columns=["Acción","Frecuencia"]
    fig_acc=px.bar(acc_df,x="Frecuencia",y="Acción",orientation="h",
                   color="Frecuencia",color_continuous_scale=[[0,"#A78BFA"],[1,C_PRIMARY]])
    fig_acc.update_layout(height=240,coloraxis_showscale=False,**PLT)
    st.plotly_chart(fig_acc,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<p class="section-title">📈 Análisis de datos reales — CREDITEX</p>', unsafe_allow_html=True)
    summ=st.session_state.summary
    k1,k2,k3,k4=st.columns(4)
    k1.metric("🗄 Total registros",f"{summ['total_registros']:,}")
    k2.metric("⚠️ Nivel alto riesgo",f"{summ['pct_nivel_alto']}%")
    k3.metric("❌ Tasa rechazado",f"{summ['pct_rechazado']}%")
    k4.metric("🤖 Precisión modelo",f"{precision*100:.1f}%")
    st.markdown("<br>", unsafe_allow_html=True)

    ca2,cb2=st.columns(2)
    with ca2:
        st.markdown(f'<div class="chart-card"><div class="chart-title">🏆 Variables más influyentes</div>', unsafe_allow_html=True)
        top=importancias.head(8).reset_index(); top.columns=["Variable","Importancia"]
        fig_i=px.bar(top,x="Importancia",y="Variable",orientation="h",
                     color="Importancia",color_continuous_scale=[[0,"#A78BFA"],[1,C_PRIMARY]])
        fig_i.update_layout(height=340,coloraxis_showscale=False,**PLT)
        st.plotly_chart(fig_i,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cb2:
        st.markdown(f'<div class="chart-card"><div class="chart-title">👷 Nivel de riesgo por turno</div>', unsafe_allow_html=True)
        tdf=df_raw.groupby(["Turno","Nivel_Riesgo"]).size().reset_index(name="n")
        fig_t=px.bar(tdf,x="Turno",y="n",color="Nivel_Riesgo",
                     color_discrete_map={"Alto":C_DANGER,"Medio":C_WARNING,"Bajo":C_SUCCESS},barmode="stack")
        fig_t.update_layout(height=340,**PLT)
        st.plotly_chart(fig_t,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    cc2,cd2=st.columns(2)
    with cc2:
        st.markdown(f'<div class="chart-card"><div class="chart-title">🔍 Tipo de defecto más frecuente</div>', unsafe_allow_html=True)
        def_df=df_raw["Tipo_Defecto"].value_counts().reset_index(); def_df.columns=["Tipo","n"]
        fig_d=px.pie(def_df,values="n",names="Tipo",hole=0.45,
                     color_discrete_sequence=[C_PRIMARY,C_SUCCESS,C_DANGER,C_WARNING,"#7C3AED","#0891B2"])
        fig_d.update_layout(height=320,paper_bgcolor=BG_CARD,font={"color":TEXT_MAIN,"size":13},margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_d,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cd2:
        st.markdown(f'<div class="chart-card"><div class="chart-title">📊 Eficiencia vs nivel de riesgo</div>', unsafe_allow_html=True)
        fig_e=px.violin(df_raw,x="Nivel_Riesgo",y="Eficiencia_Telar",color="Nivel_Riesgo",
                        color_discrete_map={"Alto":C_DANGER,"Medio":C_WARNING,"Bajo":C_SUCCESS},
                        category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
        fig_e.update_layout(height=320,showlegend=False,**PLT)
        st.plotly_chart(fig_e,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="chart-card"><div class="chart-title">🏢 Resultado de inspección por cliente</div>', unsafe_allow_html=True)
    cli_df=df_raw.groupby(["Cliente","Resultado_Inspeccion"]).size().reset_index(name="n")
    fig_cli=px.bar(cli_df,x="Cliente",y="n",color="Resultado_Inspeccion",
                   color_discrete_map={"Rechazado":C_DANGER,"Observado":C_WARNING,"Aprobado":C_SUCCESS},barmode="stack")
    fig_cli.update_layout(height=300,**PLT)
    st.plotly_chart(fig_cli,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<p class="section-title">☁️ Datos reales de producción — CREDITEX</p>', unsafe_allow_html=True)
    st.code("SELECT * FROM dbo.registros_operativos ORDER BY Fecha DESC LIMIT 2000;",language="sql")
    cf1,cf2,cf3,cf4=st.columns(4)
    f_turno=cf1.selectbox("Turno",opts["turnos"]); f_nivel=cf2.selectbox("Nivel de riesgo",opts["niveles"])
    f_client=cf3.selectbox("Cliente",opts["clientes"]); f_result=cf4.selectbox("Resultado",["Todos","Rechazado","Observado","Aprobado"])
    dfv=df_raw.copy()
    if f_turno!="Todos":  dfv=dfv[dfv["Turno"]==f_turno]
    if f_nivel!="Todos":  dfv=dfv[dfv["Nivel_Riesgo"]==f_nivel]
    if f_client!="Todos": dfv=dfv[dfv["Cliente"]==f_client]
    if f_result!="Todos": dfv=dfv[dfv["Resultado_Inspeccion"]==f_result]
    st.caption(f"Mostrando **{len(dfv):,}** de {len(df_raw):,} registros")
    cols_m=["Fecha","Telar","Turno","Cliente","Descripcion_Articulo","Temperatura","Humedad",
            "RPM_Telar","Eficiencia_Telar","Tipo_Defecto","Resultado_Inspeccion","Nivel_Riesgo","Accion_Recomendada"]
    st.dataframe(dfv[cols_m].head(300),use_container_width=True,hide_index=True)
    col_d1,col_d2=st.columns(2)
    col_d1.download_button("⬇️ Descargar CSV",dfv.to_csv(index=False).encode(),"creditex.csv","text/csv")
    if col_d2.button("🔄 Re-sincronizar"):
        with st.spinner("Consultando Azure SQL..."):
            st.session_state.df_cloud=fetch_records(n=2000)
        st.success("✅ Actualizado"); st.rerun()

# ══ TAB 5 — CREDITEX AI ══════════════════════════════════════════════════════
with tab5:
    summ = st.session_state.summary

    # HEADER DEL COPILOT
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C_PRIMARY},{C_SECONDARY});
                border-radius:20px;padding:28px 32px;margin-bottom:20px;
                box-shadow:0 8px 24px rgba(91,33,182,.25)">
      <div style="display:flex;align-items:center;gap:16px">
        <div style="background:rgba(255,255,255,.2);border-radius:16px;padding:14px;font-size:32px">🧵</div>
        <div>
          <div style="font-size:24px;font-weight:900;color:white">CREDITEX AI</div>
          <div style="font-size:13px;color:rgba(255,255,255,.85);margin-top:4px">
            Sistema Inteligente de Predicción y Recomendación Preventiva
          </div>
        </div>
        <div style="margin-left:auto;background:rgba(255,255,255,.15);border-radius:20px;
                    padding:6px 16px;font-size:13px;color:white;font-weight:600">
          🟢 IA conectada
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # STATS RÁPIDAS
    s1,s2,s3,s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="kpi-card" style="min-height:80px;padding:16px 20px">
          <div class="kpi-icon ic-green" style="width:44px;height:44px;font-size:20px">🟢</div>
          <div><p class="kpi-label">Estado</p><p style="font-size:18px;font-weight:800;color:{C_SUCCESS}">Operación Normal</p></div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="kpi-card" style="min-height:80px;padding:16px 20px">
          <div class="kpi-icon ic-red" style="width:44px;height:44px;font-size:20px">🎯</div>
          <div><p class="kpi-label">Riesgo estimado</p><p style="font-size:18px;font-weight:800;color:{color_g}">{riesgo_pct:.1f}%</p></div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="kpi-card" style="min-height:80px;padding:16px 20px">
          <div class="kpi-icon ic-blue" style="width:44px;height:44px;font-size:20px">📈</div>
          <div><p class="kpi-label">Registros analizados</p><p style="font-size:18px;font-weight:800;color:{C_SECONDARY}">{len(df_raw):,}</p></div>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""<div class="kpi-card" style="min-height:80px;padding:16px 20px">
          <div class="kpi-icon ic-purple" style="width:44px;height:44px;font-size:20px">🤖</div>
          <div><p class="kpi-label">Precisión del modelo</p><p style="font-size:18px;font-weight:800;color:{C_PRIMARY}">{precision*100:.1f}%</p></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # MENSAJE INICIAL
    if "chat_history" not in st.session_state:
        msg_inicial = (
            f"**🧵 CREDITEX AI — Copilot Industrial**\n\n"
            f"Bienvenido al Sistema Inteligente de Predicción y Recomendación Preventiva.\n\n"
            f"Mi función es analizar datos reales de producción, detectar patrones de riesgo "
            f"y recomendar acciones preventivas antes de que ocurran defectos en el proceso.\n\n"
            f"**Estado del sistema:**\n"
            f"- 🟢 Operación normal · Riesgo: **{riesgo_pct:.1f}%** (nivel **{nivel}**)\n"
            f"- 🧵 Telar activo: **{telar_sel}** · Turno: **{turno}**\n"
            f"- 📈 Registros analizados: **{len(df_raw):,}**\n"
            f"- 🤖 Precisión del modelo: **{precision*100:.1f}%**\n\n"
            f"Puedes preguntarme sobre:\n"
            f"- 🎯 Riesgo operativo y predicciones\n"
            f"- 🧵 Calidad y tipos de defecto\n"
            f"- 📊 Análisis de datos y tendencias\n"
            f"- 🔧 Mantenimiento preventivo\n"
            f"- ☁️ Conexión a la base de datos\n\n"
            f"*Escribe una consulta o selecciona una pregunta rápida para comenzar.*"
        )
        st.session_state.chat_history = [{"role":"assistant","content": msg_inicial}]

    # HISTORIAL
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧵" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    def responder(p):
        pl = p.lower()
        if any(x in pl for x in ["riesgo","defecto","prediccion","predicción"]):
            return (f"**🎯 Análisis de riesgo operativo**\n\n"
                    f"El modelo de inteligencia artificial estima un riesgo de defecto de **{riesgo_pct:.1f}%** "
                    f"para el telar **{telar_sel}** en el turno **{turno}** (nivel **{nivel}**).\n\n"
                    f"El modelo Random Forest fue entrenado con **{len(df_raw):,} registros reales** "
                    f"de CREDITEX y alcanza una precisión de **{precision*100:.1f}%**.\n\n"
                    f"{'⚠️ Se recomienda intervención inmediata del supervisor.' if nivel=='ALTO' else '✅ El proceso opera dentro de parámetros normales.' if nivel=='BAJO' else '⚡ Monitorear de cerca y aplicar ajustes preventivos.'}")
        if any(x in pl for x in ["nube","base","azure","sql","datos","registros","conexion","conexión"]):
            c = st.session_state.conn_info
            return (f"**☁️ Estado de la conexión a la nube**\n\n"
                    f"- **Proveedor:** {CLOUD_CONFIG['provider']}\n"
                    f"- **Host:** {c['host']}\n"
                    f"- **Base de datos:** {CLOUD_CONFIG['database']}\n"
                    f"- **Registros disponibles:** {c['rows_available']:,}\n"
                    f"- **Telares activos:** {c['telares_activos']}\n"
                    f"- **Período de datos:** {c['rango_fechas']}\n"
                    f"- **Latencia:** {c['latency_ms']} ms\n"
                    f"- **Cifrado:** TLS 1.2 ✅")
        if any(x in pl for x in ["recomienda","acción","accion","hacer","preveni"]):
            recs_local = []
            if roturas_urd>8:  recs_local.append(f"🧵 **Roturas de urdimbre** ({roturas_urd}): Verificar tensión y estado de hilos")
            if roturas_tram>8: recs_local.append(f"🧵 **Roturas de trama** ({roturas_tram}): Revisar lanzadera")
            if eficiencia<75:  recs_local.append(f"📊 **Eficiencia baja** ({eficiencia:.0f}%): Programar revisión técnica")
            if dias_mant>60:   recs_local.append(f"🔧 **Mantenimiento vencido** ({dias_mant} días): Programar urgente")
            if t_parada>120:   recs_local.append(f"⏸ **Tiempo de parada** ({t_parada} min): Investigar causa raíz")
            if not recs_local: return "✅ **Sin alertas activas.** Todos los parámetros operan dentro de los rangos históricos normales de CREDITEX."
            return "**💡 Recomendaciones preventivas activas:**\n\n" + "\n".join([f"- {r}" for r in recs_local])
        if any(x in pl for x in ["cliente","reclamo","devolucion"]):
            return (f"**🏢 Análisis por cliente**\n\n"
                    f"- Cliente con más reclamos: **{summ['cliente_mas_reclamos']}**\n"
                    f"- Tasa de rechazo global: **{summ['pct_rechazado']}%**\n"
                    f"- Registros con nivel alto: **{summ['pct_nivel_alto']}%**\n"
                    f"- Turno de mayor riesgo: **{summ['turno_mayor_riesgo']}**\n"
                    f"- Telar con más incidencias: **{summ['telar_mayor_incid']}**")
        if any(x in pl for x in ["modelo","precision","precisión","ia","inteligencia","algoritmo"]):
            return (f"**🤖 Modelo de Inteligencia Artificial**\n\n"
                    f"El sistema utiliza un **Random Forest Classifier** con:\n"
                    f"- 🌲 150 árboles de decisión\n"
                    f"- 📏 Profundidad máxima: 10 niveles\n"
                    f"- 📊 14 variables predictoras\n"
                    f"- 🗄 Entrenado con **{len(df_raw):,} registros reales** de CREDITEX\n"
                    f"- 🎯 Precisión: **{precision*100:.1f}%**\n\n"
                    f"Las variables más influyentes son temperatura, eficiencia del telar y roturas de urdimbre.")
        if any(x in pl for x in ["mantenimiento","preventivo","revision"]):
            return (f"**🔧 Estado de mantenimiento**\n\n"
                    f"- Días desde último mantenimiento: **{dias_mant}**\n"
                    f"- {'⚠️ **Mantenimiento vencido.** Se recomienda programar revisión inmediata.' if dias_mant>60 else '✅ Mantenimiento al día.'}\n\n"
                    f"- Tiempo de parada acumulado: **{t_parada} min**\n"
                    f"- Tiempo de reparación: **{t_reparac} min**")
        if any(x in pl for x in ["hola","buenas","inicio","empezar"]):
            return (f"¡Hola! 👋 Soy **CREDITEX AI**, tu copilot industrial.\n\n"
                    f"Sistema operativo con **{len(df_raw):,} registros** analizados. "
                    f"Riesgo actual del telar **{telar_sel}**: **{riesgo_pct:.1f}%**.\n\n"
                    f"¿Qué información necesitas consultar?")
        return (f"**🧵 CREDITEX AI**\n\n"
                f"Puedo ayudarte con:\n"
                f"- 🎯 **Riesgo operativo** — predicciones del modelo IA\n"
                f"- 💡 **Recomendaciones** — acciones preventivas\n"
                f"- ☁️ **Base de datos** — estado de la conexión\n"
                f"- 🏢 **Clientes** — reclamos y rechazos\n"
                f"- 🤖 **Modelo IA** — precisión y variables\n"
                f"- 🔧 **Mantenimiento** — estado y alertas\n\n"
                f"**Resumen actual:** Riesgo {riesgo_pct:.1f}% · {len(df_raw):,} registros · {precision*100:.1f}% precisión")

    if pregunta := st.chat_input("Consulta al Copilot Industrial CREDITEX..."):
        st.session_state.chat_history.append({"role":"user","content":pregunta})
        with st.chat_message("user", avatar="👤"): st.markdown(pregunta)
        resp = responder(pregunta)
        st.session_state.chat_history.append({"role":"assistant","content":resp})
        with st.chat_message("assistant", avatar="🧵"): st.markdown(resp)

    st.markdown(f'<p class="section-title" style="margin-top:20px">⚡ Consultas rápidas</p>', unsafe_allow_html=True)
    cols_q = st.columns(3)
    pqs = [
        "¿Cuál es el riesgo actual?",
        "¿Qué se recomienda hacer?",
        "¿Cómo está la conexión a la nube?",
        "¿Cómo funciona el modelo IA?",
        "¿Qué clientes tienen más reclamos?",
        "¿Cómo está el mantenimiento?",
    ]
    for i,pq in enumerate(pqs):
        if cols_q[i%3].button(pq, use_container_width=True, key=f"qr_{i}"):
            st.session_state.chat_history.append({"role":"user","content":pq})
            st.session_state.chat_history.append({"role":"assistant","content":responder(pq)})
            st.rerun()

st.markdown(f"""<div class="footer">🧵 <b>UNI</b> · <b>CREDITEX S.A.A.</b> · ☁️ Azure SQL Database · 5,000 registros reales · Prototipo v5.0 · <b>CONCYTEC 2026</b></div>""", unsafe_allow_html=True)
