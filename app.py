"""
Superstore Retail Sales Analytics and Profitability Prediction
Streamlit Dashboard — app.py  (Dark Enterprise Edition)
Run: streamlit run app.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Superstore Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Dark Enterprise Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
  --bg-base:      #0f1117;
  --bg-card:      #1a1d2e;
  --bg-card2:     #1e2235;
  --bg-surface:   #252840;
  --border:       #2e3250;
  --border-glow:  #3d4270;
  --accent:       #6c63ff;
  --accent2:      #00d4aa;
  --accent-warn:  #f59e0b;
  --accent-danger:#ef4444;
  --accent-ok:    #10b981;
  --text-primary: #e8eaf0;
  --text-secondary:#9aa0bc;
  --text-muted:   #5a607a;
  --radius:       14px;
  --radius-sm:    8px;
  --shadow:       0 4px 24px rgba(0,0,0,0.45);
  --shadow-sm:    0 2px 12px rgba(0,0,0,0.3);
}

/* ── Global reset ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
}

/* ── Main content area ── */
.main .block-container {
  background-color: var(--bg-base) !important;
  padding: 1.5rem 2.5rem 3rem 2.5rem !important;
  max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #12152a 0%, #0f1117 100%) !important;
  border-right: 1px solid var(--border) !important;
  padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
}

/* ── Sidebar logo block ── */
.sidebar-logo {
  background: linear-gradient(135deg, #6c63ff22 0%, #00d4aa11 100%);
  border-bottom: 1px solid var(--border);
  padding: 1.4rem 1.2rem 1.2rem 1.2rem;
  margin-bottom: 0.5rem;
}
.sidebar-logo-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 0.02em;
  margin: 0;
}
.sidebar-logo-sub {
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-top: 2px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.sidebar-badge {
  display: inline-block;
  background: var(--accent);
  color: #fff;
  font-size: 0.62rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 20px;
  margin-top: 6px;
  letter-spacing: 0.05em;
}

/* ── Sidebar nav items ── */
[data-testid="stSidebar"] .stRadio label {
  display: flex !important;
  align-items: center !important;
  padding: 0.6rem 1rem !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-secondary) !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  margin: 2px 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(108,99,255,0.12) !important;
  color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
[data-testid="stSidebar"] .stRadio input:checked + div label {
  background: linear-gradient(90deg, rgba(108,99,255,0.25) 0%, rgba(108,99,255,0.08) 100%) !important;
  color: #a99fff !important;
  border-left: 3px solid var(--accent) !important;
}

/* ── Page header ── */
.page-header {
  padding: 1.5rem 0 1rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.8rem;
}
.page-header-tag {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--accent);
  margin-bottom: 4px;
}
.page-header h1 {
  font-size: 1.75rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  margin: 0 !important;
  line-height: 1.2 !important;
}
.page-header-sub {
  font-size: 0.88rem;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* ── KPI cards ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  margin-bottom: 2rem;
}
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  cursor: default;
}
.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 32px rgba(108,99,255,0.18);
  border-color: var(--border-glow);
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-card.accent-purple::before  { background: linear-gradient(90deg, #6c63ff, #9b8fff); }
.kpi-card.accent-teal::before    { background: linear-gradient(90deg, #00d4aa, #00b8d9); }
.kpi-card.accent-amber::before   { background: linear-gradient(90deg, #f59e0b, #f97316); }
.kpi-card.accent-red::before     { background: linear-gradient(90deg, #ef4444, #f43f5e); }
.kpi-card.accent-green::before   { background: linear-gradient(90deg, #10b981, #34d399); }

.kpi-icon {
  font-size: 1.5rem;
  margin-bottom: 0.6rem;
  opacity: 0.85;
}
.kpi-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-bottom: 0.3rem;
}
.kpi-value {
  font-size: 1.85rem;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.1;
  margin-bottom: 0.3rem;
}
.kpi-delta-pos { color: var(--accent-ok);     font-size: 0.75rem; font-weight: 600; }
.kpi-delta-neg { color: var(--accent-danger);  font-size: 0.75rem; font-weight: 600; }
.kpi-desc      { color: var(--text-secondary); font-size: 0.75rem; margin-top: 2px; }

/* ── Section headers ── */
.section-header {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 1.6rem 0 0.8rem 0;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section-header-icon { font-size: 1rem; }

/* ── Chart cards ── */
.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.4rem 1rem 1.4rem;
  margin-bottom: 1rem;
  transition: border-color 0.2s ease;
}
.chart-card:hover { border-color: var(--border-glow); }
.chart-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.2rem;
}
.chart-subtitle {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-bottom: 0.8rem;
}

/* ── Insight callout ── */
.insight-box {
  background: linear-gradient(135deg, rgba(108,99,255,0.12) 0%, rgba(0,212,170,0.06) 100%);
  border: 1px solid rgba(108,99,255,0.3);
  border-left: 4px solid var(--accent);
  border-radius: var(--radius-sm);
  padding: 1rem 1.2rem;
  margin: 1rem 0;
  font-size: 0.88rem;
  color: var(--text-primary);
  line-height: 1.6;
}
.insight-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--accent);
  margin-bottom: 4px;
}

/* ── Status badges ── */
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.badge-high   { background: rgba(239,68,68,0.18);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-medium { background: rgba(245,158,11,0.18); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-low    { background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }

/* ── Result cards (prediction) ── */
.result-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.4rem 1.6rem;
  text-align: center;
  transition: border-color 0.2s;
}
.result-card:hover { border-color: var(--border-glow); }
.result-card-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                     letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.5rem; }
.result-card-value { font-size: 2.2rem; font-weight: 700; color: var(--text-primary); }
.result-card-sub   { font-size: 0.78rem; color: var(--text-secondary); margin-top: 4px; }

/* ── Probability bar ── */
.prob-track {
  background: var(--bg-surface);
  border-radius: 20px;
  height: 12px;
  overflow: hidden;
  margin: 0.6rem 0;
}
.prob-fill {
  height: 100%;
  border-radius: 20px;
  transition: width 0.6s ease;
}

/* ── Rec cards ── */
.rec-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  margin-bottom: 0.6rem;
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  transition: border-color 0.2s, transform 0.2s;
}
.rec-card:hover { border-color: var(--border-glow); transform: translateX(4px); }
.rec-priority {
  background: var(--accent);
  color: #fff;
  font-size: 0.75rem;
  font-weight: 700;
  width: 26px; height: 26px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
}
.rec-body {}
.rec-title { font-size: 0.88rem; font-weight: 600; color: var(--text-primary); }
.rec-impact { font-size: 0.78rem; color: var(--accent2); margin-top: 2px; }

/* ── Table ── */
[data-testid="stDataFrame"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] table { background: transparent !important; }
[data-testid="stDataFrame"] th {
  background: var(--bg-surface) !important;
  color: var(--text-secondary) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
  color: var(--text-primary) !important;
  font-size: 0.82rem !important;
  border-bottom: 1px solid rgba(46,50,80,0.5) !important;
}
[data-testid="stDataFrame"] tr:hover td {
  background: rgba(108,99,255,0.06) !important;
}

/* ── Form / inputs ── */
[data-testid="stForm"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1.4rem !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
  background: var(--bg-surface) !important;
  border-color: var(--border) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-sm) !important;
}
label[data-testid="stWidgetLabel"] p {
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  color: var(--text-secondary) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}

/* ── Buttons ── */
.stButton > button, .stFormSubmitButton > button {
  background: linear-gradient(135deg, #6c63ff 0%, #8b7fff 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  padding: 0.5rem 1.4rem !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.03em !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(108,99,255,0.4) !important;
}

/* ── Metrics override ── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.9rem 1.2rem !important;
}
[data-testid="stMetricLabel"] p { color: var(--text-muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.6rem !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary { color: var(--text-primary) !important; font-weight: 600 !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 3px; }

/* ── st.info / st.success / st.warning / st.error overrides ── */
[data-testid="stAlert"] {
  border-radius: var(--radius-sm) !important;
  border-left-width: 4px !important;
}

/* ── Hide default streamlit branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Caption / small text ── */
.stCaption, caption { color: var(--text-muted) !important; font-size: 0.73rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME (applied globally to all charts)
# ══════════════════════════════════════════════════════════════════════════════
CHART_BG    = "#1a1d2e"
CHART_FG    = "#e8eaf0"
CHART_GRID  = "#2e3250"
CHART_TICK  = "#9aa0bc"
ACCENT_BLUE = "#6c63ff"
ACCENT_TEAL = "#00d4aa"
ACCENT_AMBI = "#f59e0b"
ACCENT_RED  = "#ef4444"
ACCENT_GRN  = "#10b981"

plt.rcParams.update({
    "figure.facecolor":   CHART_BG,
    "axes.facecolor":     CHART_BG,
    "axes.edgecolor":     CHART_GRID,
    "axes.labelcolor":    CHART_TICK,
    "axes.titlecolor":    CHART_FG,
    "axes.grid":          True,
    "grid.color":         CHART_GRID,
    "grid.linewidth":     0.5,
    "grid.alpha":         0.6,
    "xtick.color":        CHART_TICK,
    "ytick.color":        CHART_TICK,
    "text.color":         CHART_FG,
    "legend.facecolor":   "#252840",
    "legend.edgecolor":   CHART_GRID,
    "legend.labelcolor":  CHART_FG,
    "legend.fontsize":    8,
    "figure.dpi":         130,
    "font.family":        "sans-serif",
    "font.size":          9,
})

# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def card_wrap(html: str) -> None:
    st.markdown(f'<div class="chart-card">{html}</div>', unsafe_allow_html=True)

def section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header"><span class="section-header-icon">{icon}</span>{title}</div>',
        unsafe_allow_html=True,
    )

def kpi_card(icon, label, value, desc, delta=None, delta_pos=True, accent="accent-purple"):
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-pos" if delta_pos else "kpi-delta-neg"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    return f"""
<div class="kpi-card {accent}">
  <div class="kpi-icon">{icon}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {delta_html}
  <div class="kpi-desc">{desc}</div>
</div>"""

def render_chart(fig, title="", subtitle=""):
    """Wrap a matplotlib figure in a dark chart-card."""
    hdr = ""
    if title:
        hdr += f'<div class="chart-title">{title}</div>'
    if subtitle:
        hdr += f'<div class="chart-subtitle">{subtitle}</div>'
    if hdr:
        st.markdown(f'<div class="chart-card">{hdr}', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    if hdr:
        st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig)

# ── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Superstore.csv", dtype={"Postal Code": str}, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  format="%d-%m-%Y")
    df["Ship_Lag_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Order_Year"]    = df["Order Date"].dt.year
    df["Order_Month"]   = df["Order Date"].dt.month
    df["Order_Quarter"] = df["Order Date"].dt.quarter
    df["Is_Q4"]         = df["Order Date"].dt.month.isin([10, 11, 12]).astype(int)
    df["Loss_Flag"]     = (df["Profit"] < 0).astype(int)
    return df

@st.cache_resource
def load_model():
    model     = joblib.load("models/loss_classifier.joblib")
    feat_cols = joblib.load("models/feature_columns.joblib")
    return model, feat_cols

df = load_data()
model, FEATURE_COLUMNS = load_model()

# ── Pre-compute values used in multiple tabs ──────────────────────────────────
total_sales    = df["Sales"].sum()
total_profit   = df["Profit"].sum()
loss_rate      = df["Loss_Flag"].mean() * 100
n_orders       = df["Order ID"].nunique()
n_products     = df["Product ID"].nunique()
profit_margin  = total_profit / total_sales * 100

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-title">🛒 Superstore Analytics</div>
      <div class="sidebar-logo-sub">Enterprise Intelligence Platform</div>
      <span class="sidebar-badge">4-TIER FRAMEWORK</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    tab = st.radio(
        "Navigation",
        ["📊  Descriptive Analytics",
         "🔬  Diagnostic Analytics",
         "🤖  Predictive Analytics",
         "🎯  Prescriptive Analytics"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Mini KPI summary in sidebar
    st.markdown(f"""
    <div style="padding:0.6rem 0.2rem; font-size:0.78rem; color:#9aa0bc;">
      <div style="margin-bottom:6px;">
        <span style="color:#5a607a;font-size:0.68rem;text-transform:uppercase;letter-spacing:.06em;">Dataset</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span>Records</span><strong style="color:#e8eaf0;">9,994</strong>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span>Orders</span><strong style="color:#e8eaf0;">{n_orders:,}</strong>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span>Period</span><strong style="color:#e8eaf0;">2011–2014</strong>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span>Loss Rate</span><strong style="color:#ef4444;">{loss_rate:.1f}%</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#5a607a;text-align:center;'>Random Forest · 34 features<br>ROC-AUC 98.49% · Accuracy 93.95%</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
if "Descriptive" in tab:

    # Page header
    st.markdown("""
    <div class="page-header">
      <div class="page-header-tag">Level 1</div>
      <h1>Descriptive Analytics</h1>
      <div class="page-header-sub">What happened? — Historical performance summary across all dimensions</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    kpi_html = f"""
    <div class="kpi-grid">
      {kpi_card("💰", "Total Revenue", f"${total_sales/1e6:.2f}M", "Gross sales 2011–2014",
                delta=None, accent="accent-purple")}
      {kpi_card("📈", "Net Profit", f"${total_profit/1e3:.0f}K", f"Margin: {profit_margin:.1f}%",
                delta=None, accent="accent-teal")}
      {kpi_card("⚠️", "Loss Rate", f"{loss_rate:.1f}%", f"{df['Loss_Flag'].sum():,} loss-making orders",
                delta=None, accent="accent-red")}
      {kpi_card("📦", "Unique Orders", f"{n_orders:,}", "Distinct order IDs",
                delta=None, accent="accent-amber")}
      {kpi_card("🏷️", "Products SKUs", f"{n_products:,}", "Unique product IDs",
                delta=None, accent="accent-green")}
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Row 1: Category bar + Segment donut ───────────────────────────────────
    section_header("📊", "Category & Segment Performance")
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        cat_data = df.groupby("Category")[["Sales", "Profit"]].sum().reset_index()
        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(cat_data))
        w = 0.36
        ax.bar(x - w/2, cat_data["Sales"]/1000, w, label="Sales ($K)",
               color=ACCENT_BLUE, alpha=0.9, edgecolor="none", zorder=3)
        ax.bar(x + w/2, cat_data["Profit"]/1000, w, label="Profit ($K)",
               color=ACCENT_TEAL, alpha=0.9, edgecolor="none", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_data["Category"], fontsize=9)
        ax.set_ylabel("Amount ($K)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}K"))
        ax.legend(fontsize=8, framealpha=0.3)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"${bar.get_height():.0f}K", ha="center", va="bottom",
                    fontsize=7, color=CHART_TICK)
        fig.patch.set_facecolor(CHART_BG)
        plt.tight_layout(pad=0.8)
        st.markdown('<div class="chart-card"><div class="chart-title">Sales & Profit by Category</div><div class="chart-subtitle">Three product categories across 2011–2014</div>', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close(fig)

    with col2:
        seg_data = df.groupby("Segment")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        seg_colors = [ACCENT_BLUE, ACCENT_TEAL, ACCENT_AMBI]
        wedges, texts, autos = ax.pie(
            seg_data.values, labels=seg_data.index, autopct="%1.1f%%",
            colors=seg_colors, startangle=140,
            wedgeprops=dict(width=0.62, edgecolor=CHART_BG, linewidth=2),
            pctdistance=0.78,
        )
        for t in texts:  t.set_color(CHART_TICK); t.set_fontsize(9)
        for a in autos:  a.set_color(CHART_FG);   a.set_fontsize(8); a.set_fontweight("600")
        ax.set_title("Revenue by Segment", pad=8, fontsize=9)
        fig.patch.set_facecolor(CHART_BG)
        plt.tight_layout(pad=0.8)
        st.markdown('<div class="chart-card"><div class="chart-title">Segment Revenue Distribution</div><div class="chart-subtitle">Consumer · Corporate · Home Office</div>', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close(fig)

    # ── Row 2: Monthly trend ──────────────────────────────────────────────────
    section_header("📉", "Revenue & Profit Trend")
    monthly = df.groupby(["Order_Year", "Order_Month"])[["Sales", "Profit"]].sum().reset_index()
    monthly["Period"] = pd.to_datetime(
        monthly["Order_Year"].astype(str) + "-" + monthly["Order_Month"].astype(str), format="%Y-%m")
    monthly = monthly.sort_values("Period")

    fig, ax1 = plt.subplots(figsize=(13, 3.8))
    ax2 = ax1.twinx()
    # Area under sales
    ax1.fill_between(monthly["Period"], monthly["Sales"]/1000, alpha=0.12, color=ACCENT_BLUE)
    ax1.plot(monthly["Period"], monthly["Sales"]/1000, color=ACCENT_BLUE,
             lw=2.2, label="Sales ($K)", zorder=4)
    ax2.fill_between(monthly["Period"], monthly["Profit"]/1000, alpha=0.10, color=ACCENT_TEAL)
    ax2.plot(monthly["Period"], monthly["Profit"]/1000, color=ACCENT_TEAL,
             lw=1.8, linestyle="--", label="Profit ($K)", zorder=4)
    ax2.axhline(0, color=ACCENT_RED, lw=0.8, linestyle=":", alpha=0.7)
    ax1.set_ylabel("Sales ($K)", color=ACCENT_BLUE, fontsize=8)
    ax2.set_ylabel("Profit ($K)", color=ACCENT_TEAL, fontsize=8)
    ax1.tick_params(axis="y", colors=ACCENT_BLUE)
    ax2.tick_params(axis="y", colors=ACCENT_TEAL)
    ax2.spines["right"].set_color(ACCENT_TEAL)
    ax1.spines["left"].set_color(ACCENT_BLUE)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=8,
               framealpha=0.3, facecolor=CHART_BG)
    fig.patch.set_facecolor(CHART_BG)
    plt.tight_layout(pad=0.8)
    st.markdown('<div class="chart-card"><div class="chart-title">Monthly Sales & Profit Trend (2011–2014)</div><div class="chart-subtitle">Revenue area chart with profit overlay — red dashed line marks break-even</div>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig)

    # ── Row 3: Sub-category profit horizontal bar ─────────────────────────────
    section_header("🏪", "Sub-Category Profitability")
    subcat_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values()
    bar_colors = [ACCENT_RED if v < 0 else ACCENT_GRN for v in subcat_profit.values]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.barh(subcat_profit.index, subcat_profit.values/1000,
                   color=bar_colors, edgecolor="none", height=0.65, zorder=3)
    ax.axvline(0, color=CHART_FG, lw=0.7, alpha=0.5)
    for bar, val in zip(bars, subcat_profit.values):
        x_pos = bar.get_width() + (0.5 if val >= 0 else -0.5)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"${val/1000:+.1f}K", va="center", ha=ha, fontsize=7.5, color=CHART_TICK)
    ax.set_xlabel("Total Profit ($K)", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    fig.patch.set_facecolor(CHART_BG)
    plt.tight_layout(pad=0.8)
    st.markdown('<div class="chart-card"><div class="chart-title">Total Profit by Sub-Category</div><div class="chart-subtitle">Green = net profitable  ·  Red = net loss over 2011–2014</div>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig)

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("🗂  View Raw Dataset (first 100 rows)", expanded=False):
        st.dataframe(
            df[["Order ID","Order Date","Ship Mode","Customer Name","Segment","Region",
                "Category","Sub-Category","Product Name","Sales","Quantity","Discount",
                "Profit","Loss_Flag"]].head(100),
            use_container_width=True, hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Diagnostic" in tab:

    st.markdown("""
    <div class="page-header">
      <div class="page-header-tag">Level 2</div>
      <h1>Diagnostic Analytics</h1>
      <div class="page-header-sub">Why did it happen? — Root-cause analysis of profit loss drivers</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key insight callout ───────────────────────────────────────────────────
    st.markdown("""
    <div class="insight-box">
      <div class="insight-label">⚡ Key Finding</div>
      Discounts above 20% correlate strongly with negative profit.
      At <strong>40%+ discount, 100% of transactions</strong> in this dataset result in a loss.
      The <strong>Central region</strong> shows the highest loss rate (31.9%), concentrated in
      Binders and Appliances with 80% discounts.
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Scatter + Heatmap ──────────────────────────────────────────────
    section_header("🔍", "Discount vs Profit Relationship")
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        sample_df = df.sample(min(3000, len(df)), random_state=42)
        cat_pal = {"Furniture": ACCENT_RED, "Office Supplies": ACCENT_BLUE, "Technology": ACCENT_TEAL}
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for cat, grp in sample_df.groupby("Category"):
            ax.scatter(grp["Discount"], grp["Profit"], alpha=0.28, s=11,
                       color=cat_pal.get(cat, "gray"), label=cat, zorder=3)
        for thresh, lbl in [(0.2,"20%"), (0.4,"40%"), (0.8,"80%")]:
            ax.axvline(thresh, color=ACCENT_AMBI, lw=1.1, linestyle="--", alpha=0.7, zorder=4)
            ax.text(thresh + 0.007, ax.get_ylim()[1] * 0.88,
                    f"D={lbl}", fontsize=7.5, color=ACCENT_AMBI, fontweight="600")
        ax.axhline(0, color=CHART_FG, lw=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("Discount Level", fontsize=8)
        ax.set_ylabel("Profit ($)", fontsize=8)
        ax.legend(fontsize=8, framealpha=0.3)
        fig.patch.set_facecolor(CHART_BG)
        plt.tight_layout(pad=0.8)
        st.markdown('<div class="chart-card"><div class="chart-title">Discount vs Profit by Category</div><div class="chart-subtitle">Sample of 3,000 records · amber lines = critical discount thresholds</div>', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close(fig)

    with col2:
        pivot = df.groupby(["Region", "Category"])["Loss_Flag"].mean().unstack() * 100
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        cmap = sns.diverging_palette(145, 10, as_cmap=True)
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                    linewidths=1.5, linecolor=CHART_BG, ax=ax,
                    cbar_kws={"label": "Loss Rate (%)"},
                    annot_kws={"size": 9, "weight": "bold"})
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=CHART_TICK)
        cbar.ax.tick_params(labelsize=7, colors=CHART_TICK)
        cbar.set_label("Loss Rate (%)", color=CHART_TICK, fontsize=7)
        fig.patch.set_facecolor(CHART_BG)
        plt.tight_layout(pad=0.8)
        st.markdown('<div class="chart-card"><div class="chart-title">Loss Rate Heatmap</div><div class="chart-subtitle">Region × Category — % of transactions that are losses</div>', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close(fig)

    # ── Row 2: Discount bracket analysis ─────────────────────────────────────
    section_header("📦", "Profit Distribution by Discount Bracket")
    df_diag = df.copy()
    df_diag["Discount_Bracket"] = pd.cut(df_diag["Discount"],
        bins=[-0.001, 0.001, 0.2, 0.4, 0.6, 1.0],
        labels=["0%", "1-20%", "21-40%", "41-60%", "61-80%"])

    fig, ax = plt.subplots(figsize=(12, 4.2))
    bp = df_diag.boxplot(
        column="Profit", by="Discount_Bracket", ax=ax, patch_artist=True,
        boxprops=dict(facecolor=f"{ACCENT_BLUE}33", color=ACCENT_BLUE, linewidth=1.4),
        medianprops=dict(color=ACCENT_TEAL, linewidth=2.5),
        whiskerprops=dict(color=CHART_TICK, linewidth=1.2),
        capprops=dict(color=CHART_TICK, linewidth=1.5),
        flierprops=dict(marker="o", color=CHART_TICK, alpha=0.3, markersize=3),
        return_type="dict",
    )
    ax.axhline(0, color=ACCENT_RED, linestyle=":", lw=1.2, alpha=0.8)
    ax.set_title("")
    plt.suptitle("")
    ax.set_xlabel("Discount Level", fontsize=8)
    ax.set_ylabel("Profit ($)", fontsize=8)
    fig.patch.set_facecolor(CHART_BG)
    plt.tight_layout(pad=0.8)
    st.markdown('<div class="chart-card"><div class="chart-title">Profit Distribution by Discount Level</div><div class="chart-subtitle">Box plots show median, IQR, and outliers per bracket · red dotted = break-even</div>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig)

    # ── Row 3: Pareto ─────────────────────────────────────────────────────────
    section_header("📊", "Pareto: Loss Concentration by Sub-Category")
    subcat_loss = (df[df["Loss_Flag"] == 1]
                   .groupby("Sub-Category")["Profit"].sum().abs()
                   .sort_values(ascending=False))
    cumulative = subcat_loss.cumsum() / subcat_loss.sum() * 100

    fig, ax1 = plt.subplots(figsize=(12, 4.2))
    ax2 = ax1.twinx()
    ax1.bar(subcat_loss.index, subcat_loss.values/1000, color=ACCENT_RED,
            alpha=0.85, edgecolor="none", zorder=3)
    ax2.plot(subcat_loss.index, cumulative.values, color=ACCENT_AMBI,
             marker="o", ms=5, lw=2, zorder=4)
    ax2.axhline(80, color=CHART_TICK, linestyle="--", lw=0.9, alpha=0.6)
    ax2.text(len(subcat_loss)-1, 82, "80%", fontsize=7.5, color=CHART_TICK, ha="right")
    ax1.set_ylabel("Total Loss Amount ($K)", fontsize=8, color=ACCENT_RED)
    ax2.set_ylabel("Cumulative Loss %", fontsize=8, color=ACCENT_AMBI)
    ax1.tick_params(axis="y", colors=ACCENT_RED)
    ax2.tick_params(axis="y", colors=ACCENT_AMBI)
    ax2.set_ylim(0, 110)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    fig.patch.set_facecolor(CHART_BG)
    plt.tight_layout(pad=0.8)
    st.markdown('<div class="chart-card"><div class="chart-title">Pareto Analysis — Loss by Sub-Category</div><div class="chart-subtitle">Red bars = total loss amount · amber line = cumulative % · dashed = 80% threshold</div>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig)

    # ── Discount bracket stats table ─────────────────────────────────────────
    section_header("📋", "Discount Bracket Summary")
    bracket_stats = df_diag.groupby("Discount_Bracket", observed=True)[["Profit","Loss_Flag"]].agg(
        {"Profit":"mean","Loss_Flag":"mean"}).rename(
        columns={"Profit":"Avg Profit ($)","Loss_Flag":"Loss Rate"})
    bracket_stats["Loss Rate"] = (bracket_stats["Loss Rate"]*100).map("{:.1f}%".format)
    bracket_stats["Avg Profit ($)"] = bracket_stats["Avg Profit ($)"].map("${:+.2f}".format)
    st.dataframe(bracket_stats.reset_index(), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Predictive" in tab:

    st.markdown("""
    <div class="page-header">
      <div class="page-header-tag">Level 3</div>
      <h1>Predictive Analytics</h1>
      <div class="page-header-sub">What will happen? — Random Forest classifier for real-time loss prediction</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model metric cards ────────────────────────────────────────────────────
    kpi_html = f"""
    <div class="kpi-grid">
      {kpi_card("🎯", "Accuracy",  "93.95%", "Overall correct predictions",  accent="accent-purple")}
      {kpi_card("🔬", "Precision", "83.20%", "Of predicted losses, truly losses", accent="accent-teal")}
      {kpi_card("📡", "Recall",    "84.76%", "Of actual losses, correctly caught", accent="accent-amber")}
      {kpi_card("⚖️", "F1-Score",  "83.97%", "Harmonic mean of precision & recall", accent="accent-green")}
      {kpi_card("📈", "ROC-AUC",   "98.49%", "Near-perfect discrimination ability", accent="accent-purple")}
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Model charts ──────────────────────────────────────────────────────────
    section_header("📊", "Model Evaluation Charts")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">ROC Curve</div><div class="chart-subtitle">Logistic Regression vs Random Forest</div>', unsafe_allow_html=True)
        if os.path.exists("outputs/figures/roc_curve.png"):
            st.image("outputs/figures/roc_curve.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Confusion Matrix</div><div class="chart-subtitle">Random Forest predictions on test set</div>', unsafe_allow_html=True)
        if os.path.exists("outputs/figures/confusion_matrix.png"):
            st.image("outputs/figures/confusion_matrix.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Feature Importance — Top 15 Predictors</div><div class="chart-subtitle">Discount accounts for 53.7% of model decision weight</div>', unsafe_allow_html=True)
    if os.path.exists("outputs/figures/feature_importance.png"):
        st.image("outputs/figures/feature_importance.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Live predictor ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("⚡", "Live Order Loss Predictor")
    st.markdown(
        "<p style='font-size:0.85rem;color:#9aa0bc;margin:-0.4rem 0 1rem 0;'>"
        "Enter any order parameters below to get an instant AI-powered loss probability prediction.</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        r1c1, r1c2, r1c3 = st.columns(3, gap="medium")
        with r1c1:
            ship_mode = st.selectbox("Ship Mode",
                ["Second Class", "Standard Class", "First Class", "Same Day"])
            segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
        with r1c2:
            region   = st.selectbox("Region", ["Central", "East", "South", "West"])
            category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
        with r1c3:
            subcat_map = {
                "Furniture":       ["Bookcases","Chairs","Furnishings","Tables"],
                "Office Supplies": ["Appliances","Art","Binders","Envelopes","Fasteners",
                                    "Labels","Paper","Storage","Supplies"],
                "Technology":      ["Accessories","Copiers","Machines","Phones"],
            }
            sub_category = st.selectbox("Sub-Category", subcat_map[category])

        r2c1, r2c2, r2c3, r2c4 = st.columns(4, gap="medium")
        with r2c1:
            sales    = st.number_input("Sales ($)", min_value=0.01, value=250.0, step=10.0)
        with r2c2:
            quantity = st.number_input("Quantity", min_value=1, max_value=20, value=3)
        with r2c3:
            discount = st.slider("Discount", 0.0, 0.8, 0.2, step=0.05, help="0.2 = 20% discount")
        with r2c4:
            order_year  = st.selectbox("Order Year",  [2011,2012,2013,2014], index=3)
            order_month = st.selectbox("Order Month", list(range(1,13)), index=5)

        submitted = st.form_submit_button("⚡  Run Prediction", type="primary", use_container_width=True)

    # ── Prediction result ─────────────────────────────────────────────────────
    if submitted:
        order_quarter = (order_month - 1) // 3 + 1
        is_q4 = 1 if order_month in [10,11,12] else 0
        lag_map = {"Same Day":0,"First Class":2,"Second Class":4,"Standard Class":5}
        ship_lag = lag_map.get(ship_mode, 4)

        input_dict = {col: 0.0 for col in FEATURE_COLUMNS}
        input_dict.update({
            "Sales": sales, "Quantity": quantity, "Discount": discount,
            "Ship_Lag_Days": ship_lag, "Order_Year": order_year,
            "Order_Month": order_month, "Order_Quarter": order_quarter,
            "Is_Q4": float(is_q4),
        })
        for col_key in [f"Ship Mode_{ship_mode}", f"Segment_{segment}",
                        f"Region_{region}", f"Category_{category}",
                        f"Sub-Category_{sub_category}"]:
            if col_key in input_dict:
                input_dict[col_key] = 1.0

        input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
        proba    = model.predict_proba(input_df)[0, 1]
        pred     = model.predict(input_df)[0]

        risk_tier   = "Low" if proba < 0.3 else ("Medium" if proba < 0.6 else "High")
        tier_colors = {"Low": ACCENT_GRN, "Medium": ACCENT_AMBI, "High": ACCENT_RED}
        badge_class = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}
        outcome_lbl = "LOSS PREDICTED" if pred == 1 else "PROFITABLE"
        outcome_col = ACCENT_RED if pred == 1 else ACCENT_GRN

        # Result cards
        st.markdown("<br>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3, gap="medium")
        with rc1:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-card-label">Loss Probability</div>
              <div class="result-card-value" style="color:{tier_colors[risk_tier]};">{proba*100:.1f}%</div>
              <div class="result-card-sub">Model confidence</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-card-label">Predicted Outcome</div>
              <div class="result-card-value" style="font-size:1.3rem;color:{outcome_col};">{outcome_lbl}</div>
              <div class="result-card-sub">Binary classification</div>
            </div>""", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-card-label">Risk Tier</div>
              <div class="result-card-value" style="font-size:1.5rem;">
                <span class="badge {badge_class[risk_tier]}" style="font-size:1rem;padding:6px 16px;">{risk_tier}</span>
              </div>
              <div class="result-card-sub">Threshold: Low&lt;30% · Med&lt;60% · High≥60%</div>
            </div>""", unsafe_allow_html=True)

        # Probability bar
        bar_col = tier_colors[risk_tier]
        st.markdown(f"""
        <div style="margin:1.2rem 0 0.3rem 0;">
          <div style="font-size:0.75rem;color:#9aa0bc;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">
            Loss Probability Gauge
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width:{proba*100:.1f}%;background:linear-gradient(90deg,{bar_col}99,{bar_col});"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#5a607a;margin-top:3px;">
            <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # Advisory
        advisories = {
            "High": (ACCENT_RED, "⛔ High Risk Order",
                     f"This order carries a {proba*100:.1f}% loss probability. "
                     f"Recommend reducing discount from {discount*100:.0f}% to below 20%, "
                     "or escalating for pricing review before approval."),
            "Medium": (ACCENT_AMBI, "⚠️ Medium Risk Order",
                       f"Loss probability of {proba*100:.1f}%. "
                       "Consider capping the discount and monitoring this order closely."),
            "Low": (ACCENT_GRN, "✅ Low Risk Order",
                    f"Only {proba*100:.1f}% loss probability. This order is expected to be profitable."),
        }
        col_adv, lbl_adv, msg_adv = advisories[risk_tier]
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{col_adv}18 0%,{col_adv}08 100%);
                    border:1px solid {col_adv}44;border-left:4px solid {col_adv};
                    border-radius:8px;padding:1rem 1.2rem;margin-top:1rem;">
          <div style="font-weight:700;color:{col_adv};font-size:0.88rem;margin-bottom:4px;">{lbl_adv}</div>
          <div style="font-size:0.84rem;color:#e8eaf0;">{msg_adv}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Prescriptive" in tab:

    st.markdown("""
    <div class="page-header">
      <div class="page-header-tag">Level 4</div>
      <h1>Prescriptive Analytics</h1>
      <div class="page-header-sub">What should we do? — Actionable intervention strategy to prevent financial losses</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI summary — real computed values
    kpi_html = f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
      {kpi_card("🔥", "High-Risk Exposure", "$139,939",  f"{1742:,} orders at risk",     accent="accent-red")}
      {kpi_card("💡", "Top-500 Savings",    "$125,406",  "From targeted interventions",    accent="accent-teal")}
      {kpi_card("📊", "Coverage",           "89.6%",     "Of exposure addressed by top 500", accent="accent-purple")}
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Prescriptive chart ────────────────────────────────────────────────────
    section_header("📊", "Expected Loss by Sub-Category")
    st.markdown('<div class="chart-card"><div class="chart-title">Top 10 Sub-Categories by Expected Loss Exposure</div><div class="chart-subtitle">Prioritised by model-estimated financial risk ($)</div>', unsafe_allow_html=True)
    if os.path.exists("outputs/figures/prescriptive_expected_loss.png"):
        st.image("outputs/figures/prescriptive_expected_loss.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Strategic recommendations ─────────────────────────────────────────────
    section_header("🎯", "Strategic Recommendations")
    recs = [
        ("Cap Tables discount at 30%",
         "Eliminate ~85% of Table-category losses. Tables are the single largest net-loss sub-category."),
        ("Reject Binder/Appliance orders with 80% discount in Central",
         "Save ~$21,000/year. 80% discount on these SKUs results in 100% loss rate."),
        ("Require manager approval for Discount > 40%",
         "Prevent near-certain losses — 100% loss rate observed above 40% in historical data."),
        ("Grow Technology — Copiers & Accessories",
         "Highest profit margin at low discount. Strong growth opportunity with minimal risk."),
        ("Review Central region pricing policy",
         "31.9% loss rate vs 9.9% for West. Structural discount policy reform needed."),
    ]
    for i, (title, impact) in enumerate(recs, 1):
        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-priority">{i}</div>
          <div class="rec-body">
            <div class="rec-title">{title}</div>
            <div class="rec-impact">→ {impact}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Safe discount thresholds ──────────────────────────────────────────────
    section_header("🛡️", "Safe Discount Thresholds by Sub-Category")
    col_t1, col_t2 = st.columns(2, gap="medium")
    safe_thresh = pd.DataFrame({
        "Sub-Category": ["Tables","Supplies","Binders","Appliances","Bookcases","Chairs",
                         "Machines","Phones","Copiers","Accessories","Art","Envelopes",
                         "Fasteners","Furnishings","Labels","Paper","Storage"],
        "Safe Max Discount": ["30%","10%","50%","50%","60%","50%",
                               "50%","50%","50%","50%","50%","50%",
                               "50%","50%","50%","50%","50%"],
        "Risk Level":       ["High","High","Medium","Medium","Medium","Medium",
                              "Medium","Medium","Low","Low","Low","Low",
                              "Low","Low","Low","Low","Low"],
    })
    with col_t1:
        st.dataframe(safe_thresh.head(9), use_container_width=True, hide_index=True)
    with col_t2:
        st.dataframe(safe_thresh.tail(8), use_container_width=True, hide_index=True)

    # ── Intervention priority table ───────────────────────────────────────────
    section_header("📋", "Intervention Priority List — High-Risk Orders")
    if os.path.exists("outputs/intervention_priority_list.csv"):
        interv_df = pd.read_csv("outputs/intervention_priority_list.csv")
        high_df   = interv_df[interv_df["Risk_Tier"] == "High"].head(200)

        fil1, fil2, fil3 = st.columns(3, gap="medium")
        with fil1:
            cat_filter = st.multiselect(
                "Filter by Category",
                ["All"] + sorted(high_df["Category"].unique().tolist()), default=["All"])
        with fil2:
            action_filter = st.multiselect(
                "Filter by Action",
                ["All"] + sorted(high_df["Recommended_Action"].unique().tolist()), default=["All"])
        with fil3:
            sort_col = st.selectbox("Sort by", ["Expected_Loss","Loss_Probability","Discount","Sales"])

        filtered = high_df.copy()
        if "All" not in cat_filter and cat_filter:
            filtered = filtered[filtered["Category"].isin(cat_filter)]
        if "All" not in action_filter and action_filter:
            filtered = filtered[filtered["Recommended_Action"].isin(action_filter)]
        filtered = filtered.sort_values(sort_col, ascending=False)

        st.dataframe(
            filtered[["Order ID","Category","Sub-Category","Region","Discount",
                       "Sales","Loss_Probability","Expected_Loss","Recommended_Action"]]
            .round(4),
            use_container_width=True, hide_index=True,
        )
        st.markdown(
            f"<div style='font-size:0.73rem;color:#5a607a;margin-top:4px;'>"
            f"Showing {len(filtered):,} orders · sorted by {sort_col}</div>",
            unsafe_allow_html=True,
        )
