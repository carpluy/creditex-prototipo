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
/* FONDO GENERAL */
.stApp { background: #f0f4fa !important; }
 
/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f6e 0%, #2d3494 100%) !important;
    border-right: none !important;
    min-width: 220px !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] h3 { color: #a8b8ff !important; font-size:15px !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    color: white !important; border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stButton>button {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    border-radius: 8px !important; font-weight:600 !important;
}
 
/* HEADER */
.main-header {
    background: linear-gradient(135deg, #1a1f6e 0%, #3a42c8 60%, #5b63e8 100%);
    padding: 20px 28px; border-radius: 16px;
    color: white; margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(58,66,200,0.3);
    display: flex; align-items: center; gap: 20px;
}
.header-icon {
    background: rgba(255,255,255,0.2);
    border-radius: 14px; padding: 14px;
    font-size: 28px; flex-shrink: 0;
}
.header-title { font-size:24px; font-weight:800; color:white; margin:0; }
.header-sub   { font-size:13px; color:rgba(255,255,255,0.8); margin:4px 0 0; }
.header-concytec { color:#ffd700; font-weight:700; }
 
/* TARJETAS DE MÉTRICAS — estilo colorido con iconos */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    display: flex; align-items: center; gap: 18px;
    margin-bottom: 8px;
}
.metric-icon {
    width: 56px; height: 56px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; flex-shrink: 0;
}
.metric-icon-red    { background: linear-gradient(135deg,#ff6b6b,#ee5a24); }
.metric-icon-blue   { background: linear-gradient(135deg,#74b9ff,#0984e3); }
.metric-icon-purple { background: linear-gradient(135deg,#a29bfe,#6c5ce7); }
.metric-icon-green  { background: linear-gradient(135deg,#55efc4,#00b894); }
.metric-label { font-size:13px; color:#888; font-weight:500; margin:0; }
.metric-value { font-size:32px; font-weight:800; color:#1a1f6e; margin:2px 0 0; line-height:1; }
.metric-badge {
    display:inline-block; font-size:12px; font-weight:700;
    padding:3px 10px; border-radius:20px; margin-top:6px;
}
.badge-green  { background:#d4f8e8; color:#00b894; }
.badge-red    { background:#ffe0e0; color:#ee5a24; }
.badge-yellow { background:#fff3cd; color:#e17055; }
 
/* TABS */
[data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 12px !important;
    padding: 6px !important;
    border: 1px solid #e0e8ff !important;
    gap: 4px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #666 !important;
    border-radius: 8px !important;
    font-size:14px !important; font-weight:600 !important;
    padding: 10px 20px !important; border: none !important;
}
[aria-selected="true"] {
    background: linear-gradient(135deg,#3a42c8,#5b63e8) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(58,66,200,0.4) !important;
}
 
/* EXPANDER */
[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #e0e8ff !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
 
/* ALERTAS */
.alert-danger  { background:#fff5f5;border-left:5px solid #ee5a24;padding:16px 20px;border-radius:12px;color:#c0392b;font-size:15px; }
.alert-success { background:#f0fff8;border-left:5px solid #00b894;padding:16px 20px;border-radius:12px;color:#00695c;font-size:15px; }
.alert-warning { background:#fffbf0;border-left:5px solid #fdcb6e;padding:16px 20px;border-radius:12px;color:#e17055;font-size:15px; }
.cloud-badge   { background:#f0fff8;border:1px solid #00b894;border-radius:10px;padding:12px 18px;font-size:14px;color:#00695c; }
 
/* RECOMENDACIONES */
.rec-box {
    background: white;
    border: 1px solid #e0e8ff;
    border-left: 5px solid #3a42c8;
    border-radius: 12px;
    padding: 16px 20px; margin: 10px 0;
    font-size: 15px; color: #333;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.tag-high { background:#ffe0e0;color:#ee5a24;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }
.tag-med  { background:#fff3cd;color:#e17055;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }
.tag-low  { background:#d4f8e8;color:#00b894;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700; }
 
/* TABLA DE PARÁMETROS */
.params-table { width:100%;border-collapse:collapse;font-size:15px; }
.params-table th { background:#f0f4fa;color:#1a1f6e;font-weight:700;padding:12px 16px;text-align:left; }
.params-table td { padding:11px 16px;border-bottom:1px solid #f0f0f0;color:#333; }
.params-table tr:nth-child(even) td { background:#f8faff; }
.params-table tr:hover td { background:#eef2ff; }
 
/* BOTONES */
.stButton>button {
    background: linear-gradient(135deg,#3a42c8,#5b63e8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight:700 !important;
    font-size:14px !important; padding:10px 20px !important;
    box-shadow: 0 4px 12px rgba(58,66,200,0.3) !important;
    transition: all .2s !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(58,66,200,0.5) !important;
}
 
/* CHAT */
[data-testid="stChatMessage"] {
    background: white !important;
    border: 1px solid #e0e8ff !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatInput"] textarea {
    background: white !important;
    border: 2px solid #3a42c8 !important;
    border-radius: 10px !important;
    font-size:15px !important;
}
 
/* TEXTO */
h1,h2,h3,h4 { color:#1a1f6e !important; }
.section-title { font-size:18px; font-weight:700; color:#1a1f6e; margin:16px 0 12px; }
.stMarkdown p { color:#444; font-size:15px; }
code { background:#eef2ff !important; color:#3a42c8 !important; border-radius:6px !important; }
.stCaption { color:#888 !important; font-size:13px !important; }
 
/* DATAFRAME */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #e0e8ff; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
 
/* FOOTER */
.footer { text-align:center; padding:16px; font-size:13px; color:#888; border-top:1px solid #e0e8ff; margin-top:20px; }
.footer b { color:#3a42c8; }
 
/* SCROLLBAR */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#f0f4fa; }
::-webkit-scrollbar-thumb { background:#3a42c888; border-radius:4px; }
</style>""", unsafe_allow_html=True)
 
# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="header-icon">📊</div>
  <div style="flex:1">
    <div class="header-title">Sistema Inteligente de Predicción y Recomendación Preventiva</div>
    <div class="header-sub">
      Tejido Plano — CREDITEX S.A.A. — Universidad Nacional de Ingeniería (UNI) · Datos reales de producción ·
      <span class="header-concytec">CONCYTEC 2026</span>
    </div>
  </div>
  <div style="background:rgba(255,255,255,0.15);border-radius:12px;padding:8px 14px;flex-shrink:0">
    <img src="https://www.creditex.com.pe/wp-content/uploads/2021/03/logo-creditex.png"
         height="44"
         onerror="this.outerHTML='<b style=color:white;font-size:18px>CREDITEX</b>'">
  </div>
</div>""", unsafe_allow_html=True)
 
# ── CONEXIÓN ─────────────────────────────────────────────────────────────────
if "cloud_connected" not in st.session_state:
    st.session_state.update({"cloud_connected":False,"conn_info":None,"df_cloud":None,"summary":None,"opts":None})
 
with st.expander("☁️ Conexión a base de datos en la nube — CREDITEX", expanded=not st.session_state.cloud_connected):
    col_cfg, col_btn = st.columns([3,1])
    with col_cfg:
        st.markdown(f"""<div style="display:flex;gap:24px;flex-wrap:wrap;font-size:14px;color:#555;margin:8px 0">
          <span>🖥 <b style="color:#1a1f6e">Proveedor:</b> {CLOUD_CONFIG['provider']}</span>
          <span>🌐 <b style="color:#1a1f6e">Host:</b> {CLOUD_CONFIG['host']}</span>
          <span>🗄 <b style="color:#1a1f6e">BD:</b> {CLOUD_CONFIG['database']}</span>
          <span>📍 <b style="color:#1a1f6e">Región:</b> {CLOUD_CONFIG['region']}</span>
          <span>🔒 <b style="color:#1a1f6e">SSL:</b> TLS 1.2</span>
        </div>""", unsafe_allow_html=True)
    with col_btn:
        if st.button("🔌 Conectar a la nube", use_container_width=True, type="primary"):
            with st.spinner("Estableciendo conexión segura..."):
                conn = get_connection_status()
            st.session_state.conn_info = conn
            st.session_state.cloud_connected = True
            with st.spinner(f"Descargando {conn['rows_available']:,} registros..."):
                df = fetch_records(n=2000)
            st.session_state.df_cloud = df
            with st.spinner("Calculando KPIs..."):
                st.session_state.summary = get_summary_stats()
                st.session_state.opts    = get_filter_options()
            st.success(f"✅ Conectado · {len(df):,} registros · {conn['latency_ms']} ms")
            st.rerun()
 
    if st.session_state.cloud_connected and st.session_state.conn_info:
        c = st.session_state.conn_info
        st.markdown(f"""<div class="cloud-badge">
          🟢 <b>Conectado</b> &nbsp;|&nbsp; {c['host']} &nbsp;|&nbsp;
          Latencia: <b>{c['latency_ms']} ms</b> &nbsp;|&nbsp;
          Registros: <b>{c['rows_available']:,}</b> &nbsp;|&nbsp;
          Telares: <b>{c['telares_activos']}</b> &nbsp;|&nbsp;
          Período: <b>{c['rango_fechas']}</b>
        </div>""", unsafe_allow_html=True)
 
if not st.session_state.cloud_connected:
    st.warning("⚠️ Conecta primero a la base de datos en la nube para cargar los datos reales de CREDITEX.")
    st.stop()
 
df_raw = st.session_state.df_cloud
opts   = st.session_state.opts
 
@st.cache_resource
def entrenar_modelo(df):
    d = df.copy()
    d["Turno_A"]=(d["Turno"]=="A").astype(int); d["Turno_B"]=(d["Turno"]=="B").astype(int); d["Turno_C"]=(d["Turno"]=="C").astype(int)
    d["Riesgo_bin"]=(d["Nivel_Riesgo"]=="Alto").astype(int)
    feats=["Temperatura","Humedad","RPM_Telar","Eficiencia_Telar","Tension_Urdimbre",
           "Roturas_Urdimbre","Roturas_Trama","Metros_Defectuosos","Tiempo_Parada_Min",
           "Tiempo_Reparacion_Min","Dias_Desde_Mantenimiento_Preventivo","Turno_A","Turno_B","Turno_C"]
    X=d[feats].fillna(0); y=d["Riesgo_bin"]
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=42)
    m=RandomForestClassifier(n_estimators=150,max_depth=10,random_state=42); m.fit(Xt,yt)
    acc=accuracy_score(yv,m.predict(Xv))
    imp=pd.Series(m.feature_importances_,index=feats).sort_values(ascending=False)
    return m,feats,acc,imp
 
modelo,columnas,precision,importancias=entrenar_modelo(df_raw)
 
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:14px 0 18px;border-bottom:1px solid rgba(255,255,255,0.2);margin-bottom:18px">
        <div style="background:white;border-radius:10px;padding:8px 14px;display:inline-block">
          <img src="https://www.creditex.com.pe/wp-content/uploads/2021/03/logo-creditex.png"
               height="40" onerror="this.outerHTML='<b style=color:#1a1f6e;font-size:16px>CREDITEX</b>'">
        </div>
        <div style="font-size:12px;color:rgba(255,255,255,0.7);margin-top:8px">Sistema Inteligente de Producción</div>
    </div>""", unsafe_allow_html=True)
 
    st.markdown("### ⚙️ Parámetros del telar")
    telar_sel = st.selectbox("🏭 Telar", opts["telares"][1:])
 
    if st.button("🔄 Leer último registro", use_container_width=True):
        with st.spinner(f"Consultando {telar_sel}..."):
            lec = fetch_latest_reading(telar_sel)
        if lec:
            st.session_state[f"lec_{telar_sel}"] = lec
            st.success(f"✅ Sync: {lec['timestamp']}")
 
    lec = st.session_state.get(f"lec_{telar_sel}")
    temperatura  = st.slider("🌡 Temperatura (°C)", 15.0,40.0, float(lec["temperatura"])  if lec else 25.0, 0.1)
    humedad      = st.slider("💧 Humedad (%)",      40.0,95.0, float(lec["humedad"])       if lec else 65.0, 0.5)
    rpm          = st.slider("⚙️ RPM Telar",        200,900,   int(lec["rpm_telar"])        if lec else 500,  10)
    eficiencia   = st.slider("📊 Eficiencia (%)",   60.0,100.0,float(lec["eficiencia"])    if lec else 85.0, 0.5)
    tension_urd  = st.slider("🔗 Tensión urdimbre", 15.0,50.0, 30.0, 0.5)
    roturas_urd  = st.slider("🧵 Roturas urdimbre", 0,20,3)
    roturas_tram = st.slider("🧵 Roturas trama",    0,20,3)
    metros_def   = st.slider("📏 Metros defectuosos",0,250,50)
    t_parada     = st.slider("⏸ Tiempo parada (min)",0,240,30)
    t_reparac    = st.slider("🔧 Tiempo reparación (min)",0,250,60)
    dias_mant    = st.slider("🛠 Días desde mantenimiento",0,120,30)
    turno        = st.selectbox("👷 Turno", ["A","B","C"])
 
    st.markdown("---")
    st.markdown(f"<div style='font-size:13px;color:rgba(255,255,255,0.7)'>☁️ BD: <b style='color:white'>{CLOUD_CONFIG['database']}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:13px;color:rgba(255,255,255,0.7)'>📊 {len(df_raw):,} registros cargados</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:13px;color:rgba(255,255,255,0.7)'>🤖 Precisión modelo: <b style='color:#a8b8ff'>{precision*100:.1f}%</b></div>", unsafe_allow_html=True)
 
# ── PREDICCIÓN ────────────────────────────────────────────────────────────────
ent={"Temperatura":temperatura,"Humedad":humedad,"RPM_Telar":rpm,"Eficiencia_Telar":eficiencia,
     "Tension_Urdimbre":tension_urd,"Roturas_Urdimbre":roturas_urd,"Roturas_Trama":roturas_tram,
     "Metros_Defectuosos":metros_def,"Tiempo_Parada_Min":t_parada,"Tiempo_Reparacion_Min":t_reparac,
     "Dias_Desde_Mantenimiento_Preventivo":dias_mant,
     "Turno_A":int(turno=="A"),"Turno_B":int(turno=="B"),"Turno_C":int(turno=="C")}
prob=modelo.predict_proba(pd.DataFrame([ent])[columnas])[0][1]
riesgo_pct=prob*100
if riesgo_pct>=55:   nivel,color_g,emoji,badge="ALTO","#ee5a24","🔴","badge-red"
elif riesgo_pct>=35: nivel,color_g,emoji,badge="MEDIO","#fdcb6e","🟡","badge-yellow"
else:                nivel,color_g,emoji,badge="BAJO","#00b894","🟢","badge-green"
 
PLT=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(240,244,250,0.8)",
         font={"color":"#333","size":13},
         xaxis={"gridcolor":"#dde4f0","color":"#666"},
         yaxis={"gridcolor":"#dde4f0","color":"#666"})
 
tab1,tab2,tab3,tab4,tab5=st.tabs(["📊 Monitor en tiempo real","💡 Recomendaciones","📈 Análisis","☁️ Datos de la nube","🤖 Asistente IA"])
 
# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-icon metric-icon-red">🎯</div>
          <div><p class="metric-label">Riesgo de defecto</p>
          <p class="metric-value" style="color:{color_g}">{riesgo_pct:.1f}%</p>
          <span class="metric-badge {badge}">{emoji} Nivel {nivel}</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-icon metric-icon-blue">🌡</div>
          <div><p class="metric-label">Temperatura</p>
          <p class="metric-value">{temperatura:.1f} °C</p>
          <span class="metric-badge badge-green">✔ Óptimo</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-icon metric-icon-purple">⚙️</div>
          <div><p class="metric-label">RPM Telar</p>
          <p class="metric-value">{rpm}</p>
          <span class="metric-badge badge-green">✔ Óptimo</span></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-icon metric-icon-green">📊</div>
          <div><p class="metric-label">Eficiencia</p>
          <p class="metric-value">{eficiencia:.1f}%</p>
          <span class="metric-badge {'badge-green' if eficiencia>=80 else 'badge-yellow'}">{'✔ Bueno' if eficiencia>=80 else '⚠ Revisar'}</span></div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    cg,ca=st.columns(2)
    with cg:
        st.markdown('<p class="section-title">📍 Indicador de riesgo</p>', unsafe_allow_html=True)
        fig=go.Figure(go.Indicator(
            mode="gauge+number+delta",value=riesgo_pct,
            delta={"reference":35,"suffix":"%","increasing":{"color":"#ee5a24"},"decreasing":{"color":"#00b894"}},
            title={"text":f"Riesgo de defecto<br><span style='font-size:15px;color:{color_g}'>{emoji} Nivel {nivel}</span>","font":{"color":"#1a1f6e","size":16}},
            number={"suffix":"%","font":{"size":60,"color":"#1a1f6e"}},
            gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#aaa","tickfont":{"color":"#666","size":12}},
                   "bar":{"color":color_g,"thickness":0.28},
                   "bgcolor":"white","bordercolor":"#e0e8ff","borderwidth":2,
                   "steps":[{"range":[0,35],"color":"#d4f8e8"},
                             {"range":[35,55],"color":"#fff3cd"},
                             {"range":[55,100],"color":"#ffe0e0"}],
                   "threshold":{"line":{"color":"#1a1f6e","width":3},"thickness":0.85,"value":riesgo_pct}}))
        fig.update_layout(height=340,margin=dict(t=80,b=20,l=30,r=30),
                          paper_bgcolor="white",plot_bgcolor="white",
                          font={"color":"#333"},
                          shapes=[{"type":"rect","x0":0,"y0":0,"x1":1,"y1":1,
                                   "xref":"paper","yref":"paper",
                                   "fillcolor":"white","line":{"width":0}}])
        fig.update_layout(paper_bgcolor="rgba(255,255,255,1)")
        st.plotly_chart(fig,use_container_width=True)
 
    with ca:
        st.markdown('<p class="section-title">🔔 Estado del proceso</p>', unsafe_allow_html=True)
        if nivel=="ALTO":
            st.markdown(f'<div class="alert-danger"><b>⚠️ ALERTA CRÍTICA</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b><br>Intervención inmediata del supervisor de turno.</div>',unsafe_allow_html=True)
        elif nivel=="MEDIO":
            st.markdown(f'<div class="alert-warning"><b>⚡ PRECAUCIÓN</b><br>Riesgo moderado: <b>{riesgo_pct:.1f}%</b><br>Aplicar ajustes preventivos.</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><b>✅ OPERACIÓN NORMAL</b><br>Riesgo: <b>{riesgo_pct:.1f}%</b><br>Parámetros dentro del rango.</div>',unsafe_allow_html=True)
 
        st.markdown('<p class="section-title">📋 Parámetros del telar actual</p>', unsafe_allow_html=True)
        st.markdown(f"""<table class="params-table">
          <tr><th>Parámetro</th><th>Valor</th></tr>
          <tr><td>🌡 Temperatura</td><td><b>{temperatura} °C</b></td></tr>
          <tr><td>💧 Humedad</td><td><b>{humedad} %</b></td></tr>
          <tr><td>⚙️ RPM</td><td><b>{rpm}</b></td></tr>
          <tr><td>📊 Eficiencia</td><td><b>{eficiencia} %</b></td></tr>
          <tr><td>🧵 Roturas urd.</td><td><b>{roturas_urd}</b></td></tr>
          <tr><td>🧵 Roturas tram.</td><td><b>{roturas_tram}</b></td></tr>
        </table>""", unsafe_allow_html=True)
 
    st.markdown('<p class="section-title" style="margin-top:24px">📈 Relación RPM vs Nivel de Riesgo — datos reales CREDITEX</p>', unsafe_allow_html=True)
    fig2=px.box(df_raw,x="Nivel_Riesgo",y="RPM_Telar",color="Nivel_Riesgo",
                color_discrete_map={"Alto":"#ee5a24","Medio":"#fdcb6e","Bajo":"#00b894"},
                category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
    fig2.update_layout(height=300,showlegend=False,margin=dict(t=10,b=10),
                       paper_bgcolor="white",plot_bgcolor="rgba(240,244,250,0.8)",
                       font={"color":"#333"},xaxis={"gridcolor":"#dde4f0"},yaxis={"gridcolor":"#dde4f0"})
    st.plotly_chart(fig2,use_container_width=True)
 
# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">💡 Recomendaciones preventivas del sistema experto</p>', unsafe_allow_html=True)
    recs=[]
    if roturas_urd>8:   recs.append(("🧵 Roturas de urdimbre elevadas",f"{roturas_urd} roturas. Verificar tensión y estado de hilos.","alta"))
    if roturas_tram>8:  recs.append(("🧵 Roturas de trama elevadas",f"{roturas_tram} roturas. Revisar lanzadera.","alta"))
    if eficiencia<75:   recs.append(("📊 Eficiencia baja",f"{eficiencia:.1f}%. Telares con eficiencia <75% tienen mayor tasa de rechazo.","alta"))
    if t_parada>120:    recs.append(("⏸ Tiempo de parada excesivo",f"{t_parada} min. Investigar causa raíz.","media"))
    if dias_mant>60:    recs.append(("🛠 Mantenimiento vencido",f"{dias_mant} días desde último mantenimiento. Programar urgente.","alta"))
    if metros_def>100:  recs.append(("📏 Metros defectuosos elevados",f"{metros_def} m. Activar inspección detallada.","alta"))
    if not recs:        recs.append(("✅ Sistema en condiciones óptimas","Parámetros dentro de rangos históricos normales de CREDITEX.","baja"))
 
    tag_map={"alta":"tag-high","media":"tag-med","baja":"tag-low"}
    label_map={"alta":"Prioridad Alta","media":"Prioridad Media","baja":"Prioridad Baja"}
    for titulo,texto,prio in recs:
        st.markdown(f"""<div class="rec-box">
            <b style="font-size:16px;color:#1a1f6e">{titulo}</b>
            <span class="{tag_map[prio]}" style="margin-left:12px">{label_map[prio]}</span><br>
            <span style="font-size:14px;margin-top:6px;display:block;color:#555">{texto}</span>
        </div>""",unsafe_allow_html=True)
 
    st.markdown('<p class="section-title" style="margin-top:20px">📊 Distribución de acciones recomendadas</p>', unsafe_allow_html=True)
    acc_df=df_raw["Accion_Recomendada"].value_counts().reset_index(); acc_df.columns=["Acción","Frecuencia"]
    fig_acc=px.bar(acc_df,x="Frecuencia",y="Acción",orientation="h",
                   color="Frecuencia",color_continuous_scale=[[0,"#a29bfe"],[1,"#1a1f6e"]])
    fig_acc.update_layout(height=240,coloraxis_showscale=False,margin=dict(t=10,b=10),
                          paper_bgcolor="white",plot_bgcolor="rgba(240,244,250,0.8)",
                          font={"color":"#333"},xaxis={"gridcolor":"#dde4f0"},yaxis={"gridcolor":"#dde4f0"})
    st.plotly_chart(fig_acc,use_container_width=True)
 
# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📈 Análisis de datos reales — CREDITEX</p>', unsafe_allow_html=True)
    summ=st.session_state.summary
    k1,k2,k3,k4=st.columns(4)
    with k1: st.markdown(f'<div class="metric-card"><div class="metric-icon metric-icon-blue">🗄</div><div><p class="metric-label">Total registros en BD</p><p class="metric-value" style="font-size:26px">{summ["total_registros"]:,}</p></div></div>',unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="metric-card"><div class="metric-icon metric-icon-red">⚠️</div><div><p class="metric-label">Nivel alto de riesgo</p><p class="metric-value" style="font-size:26px;color:#ee5a24">{summ["pct_nivel_alto"]}%</p></div></div>',unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="metric-card"><div class="metric-icon metric-icon-purple">❌</div><div><p class="metric-label">Tasa rechazado</p><p class="metric-value" style="font-size:26px;color:#6c5ce7">{summ["pct_rechazado"]}%</p></div></div>',unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="metric-card"><div class="metric-icon metric-icon-green">🤖</div><div><p class="metric-label">Precisión del modelo</p><p class="metric-value" style="font-size:26px;color:#00b894">{precision*100:.1f}%</p></div></div>',unsafe_allow_html=True)
 
    st.markdown("<br>",unsafe_allow_html=True)
    ca2,cb2=st.columns(2)
    with ca2:
        st.markdown('<p class="section-title">🏆 Variables más influyentes</p>',unsafe_allow_html=True)
        top=importancias.head(8).reset_index(); top.columns=["Variable","Importancia"]
        fig_i=px.bar(top,x="Importancia",y="Variable",orientation="h",
                     color="Importancia",color_continuous_scale=[[0,"#a29bfe"],[0.5,"#3a42c8"],[1,"#1a1f6e"]])
        fig_i.update_layout(height=340,coloraxis_showscale=False,margin=dict(t=10,b=10),
                            paper_bgcolor="white",plot_bgcolor="rgba(240,244,250,0.8)",
                            font={"color":"#333"},xaxis={"gridcolor":"#dde4f0"},yaxis={"gridcolor":"#dde4f0"})
        st.plotly_chart(fig_i,use_container_width=True)
    with cb2:
        st.markdown('<p class="section-title">👷 Nivel de riesgo por turno</p>',unsafe_allow_html=True)
        tdf=df_raw.groupby(["Turno","Nivel_Riesgo"]).size().reset_index(name="n")
        fig_t=px.bar(tdf,x="Turno",y="n",color="Nivel_Riesgo",
                     color_discrete_map={"Alto":"#ee5a24","Medio":"#fdcb6e","Bajo":"#00b894"},barmode="stack")
        fig_t.update_layout(height=340,margin=dict(t=10,b=10),
                            paper_bgcolor="white",plot_bgcolor="rgba(240,244,250,0.8)",
                            font={"color":"#333"},xaxis={"gridcolor":"#dde4f0"},yaxis={"gridcolor":"#dde4f0"})
        st.plotly_chart(fig_t,use_container_width=True)
 
    cc2,cd2=st.columns(2)
    with cc2:
        st.markdown('<p class="section-title">🔍 Tipo de defecto más frecuente</p>',unsafe_allow_html=True)
        def_df=df_raw["Tipo_Defecto"].value_counts().reset_index(); def_df.columns=["Tipo","n"]
        fig_d=px.pie(def_df,values="n",names="Tipo",
                     color_discrete_sequence=["#3a42c8","#00b894","#ee5a24","#fdcb6e","#a29bfe","#00cec9"])
        fig_d.update_layout(height=320,paper_bgcolor="white",font={"color":"#333","size":13},margin=dict(t=10,b=10))
        st.plotly_chart(fig_d,use_container_width=True)
    with cd2:
        st.markdown('<p class="section-title">📊 Eficiencia vs nivel de riesgo</p>',unsafe_allow_html=True)
        fig_e=px.violin(df_raw,x="Nivel_Riesgo",y="Eficiencia_Telar",color="Nivel_Riesgo",
                        color_discrete_map={"Alto":"#ee5a24","Medio":"#fdcb6e","Bajo":"#00b894"},
                        category_orders={"Nivel_Riesgo":["Bajo","Medio","Alto"]})
        fig_e.update_layout(height=320,showlegend=False,margin=dict(t=10,b=10),
                            paper_bgcolor="white",plot_bgcolor="rgba(240,244,250,0.8)",
                            font={"color":"#333"},xaxis={"gridcolor":"#dde4f0"},yaxis={"gridcolor":"#dde4f0"})
        st.plotly_chart(fig_e,use_container_width=True)
 
    st.markdown('<p class="section-title">🏢 Resultado de inspección por cliente</p>',unsafe_allow_html=True)
    cli_df=df_raw.groupby(["Cliente","Resultado_Inspeccion"]).size().reset_index(name="n")
    fig_cli=px.bar(cli_df,x="Cliente",y="n",color="Resultado_Inspeccion",
                   color_discrete_map={"Rechazado":"#ee5a24","Observado":"#fdcb6e","Aprobado":"#00b894"},barmode="stack")
    fig_cli.update_layout(height=300,margin=dict(t=10,b=10),
                          paper_bgcolor="white",plot_bgcolor="rgba(240,244,250,0.8)",
                          font={"color":"#333"},xaxis={"gridcolor":"#dde4f0"},yaxis={"gridcolor":"#dde4f0"})
    st.plotly_chart(fig_cli,use_container_width=True)
 
# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">☁️ Datos reales de producción — CREDITEX</p>',unsafe_allow_html=True)
    st.code("SELECT * FROM dbo.registros_operativos ORDER BY Fecha DESC LIMIT 2000;",language="sql")
    cf1,cf2,cf3,cf4=st.columns(4)
    f_turno =cf1.selectbox("Turno",opts["turnos"])
    f_nivel =cf2.selectbox("Nivel de riesgo",opts["niveles"])
    f_client=cf3.selectbox("Cliente",opts["clientes"])
    f_result=cf4.selectbox("Resultado",["Todos","Rechazado","Observado","Aprobado"])
    dfv=df_raw.copy()
    if f_turno!="Todos": dfv=dfv[dfv["Turno"]==f_turno]
    if f_nivel!="Todos": dfv=dfv[dfv["Nivel_Riesgo"]==f_nivel]
    if f_client!="Todos": dfv=dfv[dfv["Cliente"]==f_client]
    if f_result!="Todos": dfv=dfv[dfv["Resultado_Inspeccion"]==f_result]
    st.caption(f"Mostrando **{len(dfv):,}** de {len(df_raw):,} registros")
    cols_m=["Fecha","Telar","Turno","Cliente","Descripcion_Articulo","Temperatura","Humedad",
            "RPM_Telar","Eficiencia_Telar","Tipo_Defecto","Resultado_Inspeccion","Nivel_Riesgo","Accion_Recomendada"]
    st.dataframe(dfv[cols_m].head(300),use_container_width=True,hide_index=True)
    col_d1,col_d2=st.columns(2)
    csv=dfv.to_csv(index=False).encode("utf-8")
    col_d1.download_button("⬇️ Descargar CSV",csv,"creditex_produccion.csv","text/csv")
    if col_d2.button("🔄 Re-sincronizar"):
        with st.spinner("Consultando Azure SQL..."):
            st.session_state.df_cloud=fetch_records(n=2000)
        st.success("✅ Datos actualizados"); st.rerun()
 
# ══ TAB 5 ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">🤖 Asistente de Producción CREDITEX</p>',unsafe_allow_html=True)
    summ=st.session_state.summary
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[{"role":"assistant","content":
            f"Hola 👋 Soy el asistente del sistema predictivo de CREDITEX.\n\n"
            f"Conectado a **{CLOUD_CONFIG['database']}** con **{len(df_raw):,} registros reales**.\n\n"
            f"Riesgo actual del telar **{telar_sel}**: **{riesgo_pct:.1f}%** (nivel **{nivel}**).\n\n¿En qué te ayudo?"}]
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
 
    def responder(p):
        pl=p.lower()
        if any(x in pl for x in ["riesgo","defecto","prediccion"]): return f"Riesgo actual del telar **{telar_sel}**: **{riesgo_pct:.1f}%** (nivel **{nivel}**). Precisión del modelo: **{precision*100:.1f}%** sobre {len(df_raw):,} registros reales."
        if any(x in pl for x in ["nube","base","azure","sql","datos","registros"]): c=st.session_state.conn_info; return f"- **Proveedor:** {CLOUD_CONFIG['provider']}\n- **Registros:** {c['rows_available']:,}\n- **Período:** {c['rango_fechas']}\n- **Telares:** {c['telares_activos']}\n- **Latencia:** {c['latency_ms']} ms"
        if any(x in pl for x in ["recomienda","acción","accion","hacer"]):
            if not recs: return "✅ El proceso opera correctamente."
            r="**Recomendaciones:**\n\n"
            for tit,txt,prio in recs[:3]: r+=f"- **{tit}** ({prio}): {txt}\n\n"
            return r
        if any(x in pl for x in ["cliente","reclamo"]): return f"- Más reclamos: **{summ['cliente_mas_reclamos']}**\n- Tasa rechazo: **{summ['pct_rechazado']}%**\n- Nivel alto: **{summ['pct_nivel_alto']}%**"
        if any(x in pl for x in ["modelo","precision"]): return f"**Random Forest** (150 árboles) · {len(df_raw):,} registros reales · Precisión: **{precision*100:.1f}%**"
        if any(x in pl for x in ["hola","buenas"]): return f"¡Hola! 😊 {len(df_raw):,} registros listos. Riesgo: **{riesgo_pct:.1f}%**."
        return f"Puedo ayudarte con: riesgo, nube, recomendaciones, clientes o modelo.\n**Resumen:** {riesgo_pct:.1f}% · {len(df_raw):,} registros · {precision*100:.1f}%"
 
    if pregunta:=st.chat_input("Escribe tu consulta..."):
        st.session_state.chat_history.append({"role":"user","content":pregunta})
        with st.chat_message("user"): st.markdown(pregunta)
        resp=responder(pregunta)
        st.session_state.chat_history.append({"role":"assistant","content":resp})
        with st.chat_message("assistant"): st.markdown(resp)
 
    st.markdown('<p class="section-title" style="margin-top:16px">⚡ Preguntas rápidas:</p>',unsafe_allow_html=True)
    cols_q=st.columns(4)
    pqs=["¿Cuál es el riesgo actual?","¿Qué hay en la base de datos?","¿Qué se recomienda?","¿Cómo funciona el modelo?"]
    for i,pq in enumerate(pqs):
        if cols_q[i].button(pq,use_container_width=True):
            st.session_state.chat_history.append({"role":"user","content":pq})
            st.session_state.chat_history.append({"role":"assistant","content":responder(pq)})
            st.rerun()
 
# FOOTER
st.markdown("""<div class="footer">
    🧵 <b>UNI</b> · <b>CREDITEX S.A.A.</b> · ☁️ Azure SQL Database · 5,000 registros reales · Prototipo v3.2 · <b>CONCYTEC 2026</b>
</div>""", unsafe_allow_html=True)
