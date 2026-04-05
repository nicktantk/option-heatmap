import numpy as np
import plotly.graph_objects as go
from bs_engine import bs_price

def build_heatmaps(S, T, r,
                   sigma_min, sigma_max,
                   strike_min, strike_max,
                   n=10):
    """
    Returns a dict with meshgrid arrays for strike, sigma,
    call_price, put_price using fully vectorised NumPy operations.
    """
    sigma_vec  = np.linspace(sigma_min, sigma_max, n)
    strike_vec = np.linspace(strike_min, strike_max, n)

    # Shape: (n_sigma, n_strike) — rows = vol, cols = strike
    strike_grid, sigma_grid = np.meshgrid(strike_vec, sigma_vec)

    call_grid = bs_price(S, strike_grid, T, r, sigma_grid, option="call")
    put_grid  = bs_price(S, strike_grid, T, r, sigma_grid, option="put")

    return {
        "strike_vec":  strike_vec,
        "sigma_vec":   sigma_vec,
        "strike_grid": strike_grid,
        "sigma_grid":  sigma_grid,
        "call_grid":   call_grid,
        "put_grid":    put_grid,
    }


def _make_heatmap_trace(z, x, y, colorscale, title,
                        zmid=None, zmin=None, zmax=None,
                        text_color="rgba(255,255,255,0.88)"):
    z_rounded = np.round(z, 2)
    # Build label strings: show sign for PnL mode (when zmid==0)
    if zmid == 0:
        text_labels = np.where(
            z_rounded == 0, "0.00",
            np.where(z_rounded > 0,
                     np.char.add("+", z_rounded.astype(str)),
                     z_rounded.astype(str))
        )
    else:
        text_labels = z_rounded.astype(str)

    kwargs = dict(
        z=z_rounded,
        x=np.round(x, 2),
        y=np.round(y, 4),
        colorscale=colorscale,
        # ── in-cell annotations ──────────────────
        text=text_labels,
        texttemplate="%{text}",
        textfont=dict(
            family="IBM Plex Mono",
            size=9.5,
            color=text_color,
        ),
        # ─────────────────────────────────────────
        colorbar=dict(
            thickness=14,
            len=0.85,
            tickfont=dict(family="IBM Plex Mono", size=10, color="#64748b"),
            tickformat=".2f",
        ),
        hovertemplate=(
            "<b>" + title + "</b><br>"
            "Strike : %{x:.2f}<br>"
            "Vol    : %{y:.2%}<br>"
            "Value  : %{z:.4f}<extra></extra>"
        ),
    )
    if zmid is not None:
        kwargs["zmid"] = zmid
    if zmin is not None:
        kwargs["zmin"] = zmin
    if zmax is not None:
        kwargs["zmax"] = zmax
    return go.Heatmap(**kwargs)


def _build_single_figure(trace, title, show_yaxis_title=True):
    """
    Wrap a single heatmap trace in its own square Figure.
    Rendered inside st.columns(2): each column is ~half the page (~620px wide).
    Margins l=55, r=15, t=50, b=50  -> vertical plot area = height - 105.
    Colorbar + y-axis labels consume ~130px horizontally.
    At 620px column: heatmap width = 620 - 130 = 490px  -> height = 490 + 105 = 595.
    """
    axis_style = dict(
        showgrid=False,
        zeroline=False,
        tickfont=dict(family="IBM Plex Mono", size=10, color="#64748b"),
        title_font=dict(family="IBM Plex Mono", size=11, color="#4fc3f7"),
    )
    fig = go.Figure(trace)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="IBM Plex Mono", size=13, color="#e2e8f8"),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        margin=dict(l=55, r=15, t=50, b=55),
        height=550,
        font=dict(family="IBM Plex Sans", color="#c8d0e7"),
    )
    fig.update_xaxes(title_text="Strike Price (K)", **axis_style)
    fig.update_yaxes(
        title_text="Implied Volatility (s)" if show_yaxis_title else "",
        tickformat=".0%",
        **axis_style,
    )
    return fig


def plot_heatmaps(hm_data, pnl_mode=False,
                  call_purchase=None, put_purchase=None):
    """
    Returns (fig_left, fig_right): two independent square Plotly figures.
    Caller renders them in st.columns(2) with use_container_width=True.
    """
    strike_vec = hm_data["strike_vec"]
    sigma_vec  = hm_data["sigma_vec"]

    PNL_COLORSCALE = [
        [0.00, "#b71c1c"],
        [0.35, "#ef9a9a"],
        [0.50, "#ffffff"],
        [0.65, "#a5d6a7"],
        [1.00, "#1b5e20"],
    ]

    if pnl_mode:
        z_left  = hm_data["call_grid"] - call_purchase
        z_right = hm_data["put_grid"]  - put_purchase
        left_title  = "Call PnL"
        right_title = "Put PnL"
        abs_max_l = np.max(np.abs(z_left))
        abs_max_r = np.max(np.abs(z_right))
        trace_l = _make_heatmap_trace(z_left,  strike_vec, sigma_vec,
                                      PNL_COLORSCALE, left_title,
                                      zmid=0, zmin=-abs_max_l, zmax=abs_max_l,
                                      text_color="rgba(15,21,37,0.85)")
        trace_r = _make_heatmap_trace(z_right, strike_vec, sigma_vec,
                                      PNL_COLORSCALE, right_title,
                                      zmid=0, zmin=-abs_max_r, zmax=abs_max_r,
                                      text_color="rgba(15,21,37,0.85)")
    else:
        z_left  = hm_data["call_grid"]
        z_right = hm_data["put_grid"]
        left_title  = "Call Price"
        right_title = "Put Price"
        trace_l = _make_heatmap_trace(z_left,  strike_vec, sigma_vec,
                                      "Viridis", left_title,
                                      text_color="rgba(255,255,255,0.88)")
        trace_r = _make_heatmap_trace(z_right, strike_vec, sigma_vec,
                                      "Viridis", right_title,
                                      text_color="rgba(255,255,255,0.88)")

    fig_l = _build_single_figure(trace_l, left_title,  show_yaxis_title=True)
    fig_r = _build_single_figure(trace_r, right_title, show_yaxis_title=True)
    return fig_l, fig_r

