"""
Black-Scholes Option Pricing & Risk Analysis Dashboard
A high-performance Streamlit application for quantitative finance.
"""

import streamlit as st
import plotly.graph_objects as go
from bs_engine import bs_price, bs_greeks
from heatmap_builder import build_heatmaps, plot_heatmaps
# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BS Option Pricer",
    page_icon="BS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — dark quant terminal aesthetic
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    /* ── global ── */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0a0e1a;
        color: #c8d0e7;
    }

    /* ── sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f1525;
        border-right: 1px solid #1e2d50;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #4fc3f7;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
    }

    /* ── main area ── */
    .main .block-container { padding: 1.5rem 2rem; }

    /* ── section headers ── */
    h1 { font-family: 'IBM Plex Mono', monospace; color: #e2e8f8 !important; letter-spacing: -0.02em; }
    h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #4fc3f7 !important; }

    /* ── metric cards ── */
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 14px 18px;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover { border-color: #4fc3f7; }
    [data-testid="stMetricLabel"]  { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
    [data-testid="stMetricValue"]  { color: #e2e8f8 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem !important; }

    /* ── divider ── */
    hr { border-color: #1e2d50 !important; }

    /* ── number inputs & sliders ── */
    input[type="number"] {
        background: #0a0e1a !important;
        color: #c8d0e7 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 4px;
    }

    /* ── caption / subtitle ── */
    .subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #4fc3f7;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .badge {
        display: inline-block;
        background: #0d2137;
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        color: #4fc3f7;
        margin-left: 6px;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Model Inputs")

    st.markdown("### 📊 Market Data")
    S     = st.number_input("Underlying Price (S)",      min_value=1.0,   value=100.0, step=1.0,   format="%.2f")
    K     = st.number_input("Strike Price (K)",          min_value=1.0,   value=100.0, step=1.0,   format="%.2f")
    T     = st.number_input("Time to Expiration (T, yr)",min_value=0.001, value=1.0,   step=0.01,  format="%.3f")
    r     = st.number_input("Risk-Free Rate (r)",        min_value=0.0,   value=0.05,  step=0.001, format="%.4f")
    sigma = st.number_input("Implied Volatility (σ)",    min_value=0.001, value=0.20,  step=0.01,  format="%.4f")

    st.markdown("---")
    st.markdown("### 🗺️ Heatmap Range")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        vol_min = st.number_input("Vol Min", min_value=0.01, max_value=0.99, value=0.10, step=0.01, format="%.2f")
    with col_v2:
        vol_max = st.number_input("Vol Max", min_value=0.01, max_value=2.00, value=0.50, step=0.01, format="%.2f")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        k_min = st.number_input("Strike Min", min_value=1.0, value=max(1.0, S * 0.80), step=1.0, format="%.1f")
    with col_k2:
        k_max = st.number_input("Strike Max", min_value=1.0, value=S * 1.20, step=1.0, format="%.1f")

    st.markdown("---")
    st.markdown("### 💰 PnL Analysis")
    pnl_mode = st.checkbox("Calculate PnL", value=False)
    call_purchase = put_purchase = None
    if pnl_mode:
        call_purchase = st.number_input("Call Purchase Price", min_value=0.0, value=5.0, step=0.01, format="%.4f")
        put_purchase  = st.number_input("Put Purchase Price",  min_value=0.0, value=5.0, step=0.01, format="%.4f")

    st.markdown("---")
    st.caption("Model: Black-Scholes-Merton (1973)")
    st.caption("Greeks: Analytical closed-form")


# ─────────────────────────────────────────────
#  COMPUTE POINT ESTIMATES
# ─────────────────────────────────────────────

call_price = float(bs_price(S, K, T, r, sigma, "call"))
put_price  = float(bs_price(S, K, T, r, sigma, "put"))
greeks     = bs_greeks(S, K, T, r, sigma)

call_delta = float(greeks["call_delta"])
put_delta  = float(greeks["put_delta"])
gamma      = float(greeks["gamma"])
call_theta = float(greeks["call_theta"])
put_theta  = float(greeks["put_theta"])


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────

st.markdown(
    """
    <h1 style='margin-bottom:0'>
        Black-Scholes Option Pricer
        <span class='badge'>BSM 1973</span>
    </h1>
    <p class='subtitle'>European Options · Analytical Greeks · Interactive Risk Surface</p>
    <hr/>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  GREEKS DASHBOARD
# ─────────────────────────────────────────────

st.markdown("### Greeks Dashboard")

# ── Row 1 : Price cards ───────────────────────────────────────────────────
price_col1, price_col2 = st.columns(2)

with price_col1:
    st.markdown(
        f"""
        <div style="
            background:#0a2318;
            border:1px solid #166534;
            border-radius:8px;
            padding:16px 20px;
            text-align:center;
        ">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                        color:#4ade80;text-transform:uppercase;letter-spacing:0.1em;
                        margin-bottom:6px;">
                Call Price
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;
                        color:#86efac;font-weight:600;">
                ${call_price:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with price_col2:
    st.markdown(
        f"""
        <div style="
            background:#2d0a0a;
            border:1px solid #991b1b;
            border-radius:8px;
            padding:16px 20px;
            text-align:center;
        ">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                        color:#f87171;text-transform:uppercase;letter-spacing:0.1em;
                        margin-bottom:6px;">
                Put Price
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;
                        color:#fca5a5;font-weight:600;">
                ${put_price:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

# ── Row 2 : Greeks ────────────────────────────────────────────────────────
g1, g2, g3, g4, g5 = st.columns(5)

with g1:
    st.metric("Δ Call Delta", f"{call_delta:.4f}")
with g2:
    st.metric("Δ Put Delta",  f"{put_delta:.4f}")
with g3:
    st.metric("Γ Gamma",      f"{gamma:.6f}")
with g4:
    st.metric("Θ Call Theta", f"{call_theta:.4f}")
with g5:
    st.metric("Θ Put Theta",  f"{put_theta:.4f}")

st.markdown(
    """
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.7rem; color:#334155;
                margin-top:6px; padding:6px 10px; background:#0f1525;
                border:1px solid #1e2d50; border-radius:6px; line-height:1.8'>
        &nbsp;S={S:.2f} &nbsp;|&nbsp; K={K:.2f} &nbsp;|&nbsp;
        T={T:.3f}y &nbsp;|&nbsp; r={r:.2%} &nbsp;|&nbsp; σ={sigma:.2%}
        &nbsp;|&nbsp; Put-Call parity: C−P = {pcp:+.4f}
    </div>
    """.format(
        S=S, K=K, T=T, r=r, sigma=sigma,
        pcp=call_price - put_price,
    ),
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEATMAP SECTION
# ─────────────────────────────────────────────

vol_min_safe   = min(vol_min, vol_max - 0.01)
vol_max_safe   = max(vol_max, vol_min + 0.01)
strike_min_safe = min(k_min, k_max - 1.0)
strike_max_safe = max(k_max, k_min + 1.0)

hm_data = build_heatmaps(
    S=S, T=T, r=r,
    sigma_min=vol_min_safe,
    sigma_max=vol_max_safe,
    strike_min=strike_min_safe,
    strike_max=strike_max_safe,
    n=10,
)

if pnl_mode:
    st.markdown("### PnL Surface  _(diverging: Profit / Loss)_")
else:
    st.markdown("### Option Price Surface")

fig_l, fig_r = plot_heatmaps(
    hm_data,
    pnl_mode=pnl_mode,
    call_purchase=call_purchase,
    put_purchase=put_purchase,
)
hm_col1, hm_col2 = st.columns(2, gap="small")
with hm_col1:
    st.plotly_chart(fig_l, use_container_width=True)
with hm_col2:
    st.plotly_chart(fig_r, use_container_width=True)



# ─────────────────────────────────────────────
#  FORMULA REFERENCE EXPANDER
# ─────────────────────────────────────────────

with st.expander("Formula Reference"):
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("**Black-Scholes-Merton Prices**")
        st.latex(r"C = S\,N(d_1) - K e^{-rT} N(d_2)")
        st.latex(r"P = K e^{-rT} N(-d_2) - S\,N(-d_1)")
        st.latex(r"d_1 = \frac{\ln(S/K)+(r+\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")

    with col_f2:
        st.markdown("**Greeks**")
        st.latex(r"\Delta_{\text{call}} = N(d_1), \quad \Delta_{\text{put}} = N(d_1)-1")
        st.latex(r"\Gamma = \frac{N'(d_1)}{S\,\sigma\sqrt{T}}")
        st.latex(
            r"\Theta_{\text{call}} = -\frac{S\,N'(d_1)\sigma}{2\sqrt{T}}"
            r"- r K e^{-rT} N(d_2)"
        )
        st.latex(
            r"\Theta_{\text{put}}  = -\frac{S\,N'(d_1)\sigma}{2\sqrt{T}}"
            r"+ r K e^{-rT} N(-d_2)"
        )

    st.markdown(
        """
        <small style='color:#334155; font-family:"IBM Plex Mono",monospace'>
        N(·) = standard normal CDF &nbsp;·&nbsp;
        N′(·) = standard normal PDF &nbsp;·&nbsp;
        Theta is expressed per calendar day (÷365)
        </small>
        """,
        unsafe_allow_html=True,
    )