"""OceanEmbed UI kit — dark ocean glassmorphism theme, animations, charts.

Kept dependency-light (streamlit + plotly only) so the demo stays a thin presentation
layer over the trained artifacts. All visuals degrade gracefully without each other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- palette
INK          = "#020c1a"
PANEL        = "rgba(13, 32, 54, 0.55)"
PANEL_SOLID  = "#0c2035"
BORDER       = "rgba(103, 232, 249, 0.16)"
TEXT         = "#DCEBFF"
TEXT_MUTED   = "#8FB1CF"
CYAN         = "#22d3ee"
TEAL         = "#2dd4bf"
OCEAN        = "#0ea5e9"
VIOLET       = "#818cf8"
GOOD         = "#34d399"
WARM         = "#fbbf24"
HOT          = "#f87171"
WAVE_SPEED  = "26s"

# --------------------------------------------------------------------------- CSS
_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {{
  --cyan: {CYAN}; --teal: {TEAL}; --ocean: {OCEAN}; --violet: {VIOLET};
  --good: {GOOD}; --warm: {WARM}; --hot: {HOT};
  --panel: {PANEL}; --border: {BORDER}; --text: {TEXT}; --muted: {TEXT_MUTED};
}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: radial-gradient(1200px 800px at 15% -10%, #0b2a4a 0%, transparent 55%),
              radial-gradient(1000px 700px at 110% 10%, #062a3a 0%, transparent 50%),
              radial-gradient(900px 900px at 50% 120%, #0f1d3d 0%, transparent 55%),
              linear-gradient(160deg, #020a16 0%, #041426 55%, #04101f 100%);
  color: {TEXT};
  font-family: 'Inter', system-ui, sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}

/*** animated background orbs ***/
.oe-globe {{
  position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;
}}
.oe-globe::before, .oe-globe::after {{
  content: ""; position: absolute; border-radius: 50%; filter: blur(90px); opacity: .22;
  animation: globeDrift 22s ease-in-out infinite;
}}
.oe-globe::before {{ width: 560px; height: 560px; background: radial-gradient(circle, {OCEAN}, transparent 70%);
  top: -120px; right: -60px; }}
.oe-globe::after {{ width: 640px; height: 640px; background: radial-gradient(circle, {VIOLET}, transparent 70%);
  bottom: -180px; left: -120px; animation-delay: -11s; }}
@keyframes globeDrift {{ 0%,100% {{ transform: translate(0,0) scale(1); }}
  50% {{ transform: translate(30px,-24px) scale(1.08); }} }}

/*** hero ***/
.oe-hero {{ position: relative; z-index: 1; text-align: center; padding: .2rem 0 .2rem; }}
.oe-hero h1 {{
  font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 3.2rem; margin: 0;
  letter-spacing: -0.02em; cursor: default;
  background: linear-gradient(100deg, {CYAN}, {TEAL}, {OCEAN}, {VIOLET}, {CYAN});
  background-size: 300% 300%;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: heroShimmer 8s ease-in-out infinite;
}}
@keyframes heroShimmer {{ 0%,100% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} }}
.oe-hero p {{ color: {TEXT_MUTED}; margin: .45rem auto 0; max-width: 660px; line-height: 1.5;
  font-size: 1.02rem; }}
.oe-pulse-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
  background: {TEAL}; margin-right:.5rem; position:relative; }}
.oe-pulse-dot::after {{ content:""; position:absolute; inset:0; border-radius:50%;
  background:{TEAL}; animation: dotPulse 1.8s ease-out infinite; }}
@keyframes dotPulse {{ 0% {{ transform: scale(1); opacity:.9; }} 100% {{ transform: scale(3); opacity:0; }} }}

/*** native metric strip (no custom markup in the DOM) ***/
[data-testid="stMetric"] {{
  border-radius: 16px; padding: 1rem .9rem .8rem; height: 100%;
  background: linear-gradient(180deg, rgba(13,32,54,.55), rgba(8,24,44,.55));
  border: 1px solid {BORDER};
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 16px 34px -22px rgba(2,12,26,.9);
  transition: transform .3s cubic-bezier(.22,1,.36,1), border-color .3s, box-shadow .3s;
  animation: cardRise .7s cubic-bezier(.22,1,.36,1) both;
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-4px);
  border-color: rgba(103,232,249,.45); box-shadow: 0 22px 44px -22px rgba(34,211,238,.35); }}
[data-testid="stMetricValue"] {{
  color: {CYAN}; font-family: 'Outfit'; font-weight: 700;
  text-shadow: 0 0 18px rgba(34,211,238,.35);
}}
[data-testid="stMetricDelta"] {{ display:none; }}

/*** section header ***/
.oe-sec {{ position:relative; z-index:1; display:flex; align-items:center; gap:.6rem;
  margin: .4rem 0 .45rem; }}
.oe-sec .tick {{ width:5px; height:20px; border-radius:3px;
  background:linear-gradient(180deg,{CYAN},{TEAL}); box-shadow:0 0 14px {CYAN}; }}
.oe-sec h3 {{ font-family:'Outfit'; font-weight:700; font-size:1.12rem; margin:0;
  letter-spacing:-.01em; }}
.oe-sec .meta {{ margin-left:auto; color:{TEXT_MUTED}; font-size:.78rem; }}

/*** glass panel wrapper ***/
.oe-panel {{
  position: relative; z-index: 1; border-radius: 22px; padding: .9rem 1rem 1.1rem;
  background: {PANEL}; border: 1px solid {BORDER}; backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow: 0 24px 60px -30px rgba(2,12,26,.9);
  animation: cardRise .8s cubic-bezier(.22,1,.36,1) both;
}}
@keyframes cardRise {{ from {{ opacity:0; transform: translateY(16px) scale(.985); }}
  to {{ opacity:1; transform: translateY(0) scale(1); }} }}

/*** map frame ***/
.oe-map-frame {{ border-radius: 18px; overflow: hidden; border: 1px solid {BORDER};
  background:#04101f; transition: box-shadow .4s; }}
.oe-map-frame:has(iframe:hover) {{ box-shadow: 0 0 0 1px rgba(103,232,249,.35), 0 30px 60px -30px rgba(34,211,238,.35); }}
iframe[title="streamlit_folium.st_folium"] {{ border-radius: 18px !important; }}
.stDeployButton, [data-testid="stDecoration"] {{ display:none; }}

/*** prediction panel ***/
.oe-pred {{
  position: relative; z-index: 1; margin-top: 1rem; border-radius: 22px; padding: 1.1rem 1.15rem 1.25rem;
  background: linear-gradient(180deg, rgba(13,38,64,.92), rgba(8,24,44,.92));
  border: 1px solid rgba(34,211,238,.28);
  box-shadow: 0 0 0 1px rgba(2,12,26,.4) inset, 0 30px 70px -30px rgba(34,211,238,.28);
  animation: predIn .75s cubic-bezier(.22,1,.36,1) both;
}}
@keyframes predIn {{ 0% {{ opacity:0; transform: translateY(22px) scale(.98); }}
  100% {{ opacity:1; transform: translateY(0) scale(1); }} }}
.oe-pred .glowline {{ height:1px; margin:0 0 1rem; border:none;
  background:linear-gradient(90deg, transparent, {CYAN}, transparent); }}

/*** chips ***/
.oe-chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
.oe-chip {{
  border-radius: 12px; padding: .45rem .7rem; min-width: 92px; text-align:center;
  background: rgba(56,189,248,.08); border:1px solid {BORDER};
  transition: transform .25s, border-color .25s, box-shadow .25s;
}}
.oe-chip:hover {{ transform: translateY(-2px); border-color: rgba(103,232,249,.5);
  box-shadow: 0 10px 24px -14px rgba(34,211,238,.5); }}
.oe-chip b {{ display:block; font-family:'Outfit'; font-size:1.05rem; color:{CYAN}; }}
.oe-chip span {{ font-size:.74rem; color:{TEXT_MUTED}; }}

/*** value bars (surface features) ***/
.oe-barlist {{ display:flex; flex-direction:column; gap:9px; }}
.oe-bar {{ display:flex; align-items:center; gap:10px; }}
.oe-bar label {{ width:74px; font-size:.76rem; color:{TEXT_MUTED}; }}
.oe-bar .track {{ flex:1; height:9px; border-radius:6px; background:rgba(18,40,66,.7);
  overflow:hidden; box-shadow: inset 0 0 0 1px rgba(103,232,249,.08); }}
.oe-bar .fill {{ height:100%; border-radius:6px; width:0%;
  background:linear-gradient(90deg,{CYAN},{TEAL}); box-shadow:0 0 12px {CYAN};
  transition: width 1.1s cubic-bezier(.22,1,.36,1); }}
.oe-bar .val {{ width:58px; text-align:right; font-family:'Outfit'; font-size:.84rem; color:{TEXT}; }}

/*** sidebar ***/
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #06182b 0%, #04101f 100%);
  border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
.oe-side-title {{ font-family:'Outfit'; font-weight:700; letter-spacing:.03em; color:{CYAN};
  font-size:1.02rem; display:flex; align-items:center; gap:.45rem; }}

.oe-pill {{ display:inline-flex; align-items:center; gap:.42rem; margin-top:.3rem;
  border-radius: 999px; padding:.24rem .66rem; font-size:.74rem; font-weight:600; }}
.oe-pill[title] {{ cursor: help; }}
.oe-pill.live   {{ color:{GOOD}; background:rgba(52,211,153,.1); border:1px solid rgba(52,211,153,.35); }}
.oe-pill.cached {{ color:{CYAN}; background:rgba(34,211,238,.08); border:1px solid rgba(34,211,238,.32); }}
.oe-pill.sample {{ color:{WARM}; background:rgba(251,191,36,.1); border:1px solid rgba(251,191,36,.38); }}
.oe-pill.fail   {{ color:{HOT}; background:rgba(248,113,113,.1); border:1px solid rgba(248,113,113,.38); }}
.oe-pill.off    {{ color:{TEXT_MUTED}; background:rgba(143,177,207,.08); border:1px solid rgba(143,177,207,.2); }}
.oe-pill .dot {{ width:7px; height:7px; border-radius:50%; }}
.oe-pill.live .dot {{ background:{GOOD}; animation: dotPulse 1.8s ease-out infinite; }}
.oe-pill.cached .dot {{ background:{CYAN}; }}
.oe-pill.sample .dot {{ background:{WARM}; }}
.oe-pill.fail .dot {{ background:{HOT}; }}
.oe-pill.off .dot {{ background:{TEXT_MUTED}; }}

.oe-source {{ display:flex; flex-direction:column; padding:.55rem .65rem; margin-top:.4rem;
  border-radius: 12px; background: rgba(13,32,54,.45); border:1px solid {BORDER}; }}
.oe-source[title] {{ cursor: help; }}
.oe-source .row {{ display:flex; justify-content:space-between; align-items:center; }}

/*** data legend + banners + demo chips ***/
.oe-legend {{ display:flex; flex-direction:column; gap:5px; }}
.oe-leg-row {{ display:flex; align-items:center; gap:.5rem; font-size:.76rem; color:{TEXT_MUTED};
  line-height:1; }}
.oe-leg-dot {{ width:9px; height:9px; border-radius:50%; flex:none;
  box-shadow:0 0 8px rgba(255,255,255,.22); }}

/*** anomaly banner (profile view) ***/
.oe-banner {{ position:relative; z-index:1; display:flex; align-items:center; gap:.65rem;
  margin:.35rem 0 .7rem; border-radius:14px; padding:.7rem .9rem; font-size:.92rem; font-weight:600;
  backdrop-filter: blur(10px); animation: cardRise .6s cubic-bezier(.22,1,.36,1) both; }}
.oe-banner.warn {{ color:#fecdd3; background:rgba(248,113,113,.13);
  border:1px solid rgba(248,113,113,.5); box-shadow:0 0 30px -12px rgba(248,113,113,.65); }}
.oe-banner.ok {{ color:{GOOD}; background:rgba(52,211,153,.08); border:1px solid rgba(52,211,153,.4);
  box-shadow:0 0 26px -14px rgba(52,211,153,.5); }}
.oe-banner .dot {{ flex:none; width:10px; height:10px; border-radius:50%; }}
.oe-banner.warn .dot {{ background:{HOT}; box-shadow:0 0 12px {HOT}; animation:dotPulse 1.8s ease-out infinite; }}
.oe-banner.ok .dot {{ background:{GOOD}; box-shadow:0 0 10px {GOOD}; }}

/*** demo location quick-select ***/
.oe-demo-note {{ position:relative; z-index:1; margin-top:.2rem; font-size:.76rem; color:{TEXT_MUTED}; }}

/*** trust badge ***/
.oe-trust {{ display:inline-flex; align-items:center; gap:.45rem; margin-top:.55rem;
  border-radius: 999px; padding:.3rem .9rem; font-size:.8rem; font-weight:600;
  color: {GOOD}; background: rgba(52,211,153,.1); border: 1px solid rgba(52,211,153,.42);
  box-shadow: 0 0 24px -10px rgba(52,211,153,.55); }}
.oe-trust .dot {{ width:8px; height:8px; border-radius:50%; background: {GOOD};
  box-shadow: 0 0 10px {GOOD}; }}

.stButton > button {{
  border-radius: 12px; border:1px solid {BORDER}; background:rgba(34,211,238,.08);
  color:{CYAN}; font-weight:600; transition: all .3s cubic-bezier(.22,1,.36,1);
}}
.stButton > button:hover {{ background:rgba(34,211,238,.18); border-color:{CYAN};
  box-shadow: 0 8px 24px -10px {CYAN}; transform: translateY(-1px); }}

[data-testid="stStatusWidget"] {{ background: rgba(13,32,54,.6); border:1px solid {BORDER};
  border-radius: 12px; backdrop-filter: blur(8px); }}

.stTabs [data-baseweb="tab"] {{ color:{TEXT_MUTED}; transition: color .25s; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg,{CYAN},{TEAL}); }}

/*** animated waves + rising bubbles (cartoon ocean) ***/
.oe-waves {{ position: relative; z-index: 1; height: 64px; margin: -6px 0 -20px;
  overflow: hidden; pointer-events: none; }}
.oe-waves svg {{ position: absolute; bottom: -14px; height: 74px; width: 200%;
  animation: waveSlide {WAVE_SPEED} linear infinite; }}
.oe-waves svg:nth-child(2) {{ animation-duration: 58s; opacity: .55; }}
@keyframes waveSlide {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}

/*** footer ***/
.oe-foot {{ position:relative; z-index:1; text-align:center; margin-top: 1.4rem;
  padding-top: 1rem; color:{TEXT_MUTED}; font-size:.74rem;
  border-top: 1px solid rgba(103,232,249,.1); line-height:1.6; }}
.oe-foot b {{ color:{CYAN}; }}
"""


# --------------------------------------------------------------------------- helpers
def inject_theme() -> None:
    """Apply the global dark-ocean theme + water layers."""
    st.markdown('<div class="oe-globe"></div>', unsafe_allow_html=True)
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, framing: str | None = None) -> None:
    pitch = (
        f'<p style="color:{TEXT};font-weight:500;font-size:1.05rem;margin:.8rem auto 0;'
        f'max-width:820px;line-height:1.55;letter-spacing:.01em">{framing}</p>'
        if framing else ""
    )
    st.markdown(
        f"""
        <div class="oe-hero">
          <h1>{title}</h1>
          {pitch}
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def trust_badge(text: str) -> None:
    st.markdown(
        f'<div style="text-align:center"><span class="oe-trust"><span class="dot"></span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def mode_badge(text: str, *, kind: str = "cached", tooltip: str = "") -> None:
    """Centred data-mode badge shown below the hero.

    ``kind`` maps to a pill style: "live" (real-time fetch), "cached" / "cached_fallback"
    (offline snapshot), "hybrid" (some sources live, some cached), or "degraded".
    Tooltip explains what the mode means.
    """
    cls = {"cached": "cached", "live": "live", "hybrid": "live",
           "cached_fallback": "cached", "degraded": "sample"}.get(kind, "cached")
    tip = f' title="{_html_escape(tooltip)}"' if tooltip else ""
    st.markdown(
        f'<div style="text-align:center;margin:.2rem 0 .1rem">'
        f'<span class="oe-pill {cls}"{tip}><span class="dot"></span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def anomaly_banner(message: str, kind: str) -> None:
    """Prominent anomaly / all-clear banner above the profile chart.

    ``kind``: "warn" (possible marine heatwave / anomaly) or "ok" (within normal range).
    """
    cls = "warn" if kind == "warn" else "ok"
    st.markdown(
        f'<div class="oe-banner {cls}"><span class="dot"></span>{message}</div>',
        unsafe_allow_html=True,
    )


def legend_html(items: list[tuple[str, str]]) -> str:
    """Shared data legend: ``(label, color)`` rows rendered as dot + text.

    Used identically by the map view (sidebar) and the prediction/profile view so a
    single change here fixes the legend everywhere.
    """
    rows = "".join(
        f'<div class="oe-leg-row"><span class="oe-leg-dot" style="background:{c}"></span>'
        f"<span>{label}</span></div>"
        for label, c in items
    )
    return f'<div class="oe-legend">{rows}</div>'


def _html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def stat_metrics(cells: int, items: list[dict[str, Any]]) -> None:
    """Native Streamlit metric strip styled to match the ocean theme.

    ``items``: [{value, label, delta, help}] — rendered with st.metric inside
    st.columns, so no custom HTML markup ever reaches the page.
    """
    cols = st.columns(min(cells, max(1, len(items))))
    for col, it in zip(cols, items):
        with col:
            st.metric(it["label"], it["value"], it.get("delta", None), help=it.get("help"))


_LAKE = (
    'M0,40 C120,8 240,72 360,40 C480,8 600,72 720,40 '
    'C840,8 960,72 1080,40 C1200,8 1320,72 1440,40 '
    'C1560,8 1680,72 1800,40 C1920,8 2040,72 2160,40 '
    'C2280,8 2400,72 2520,40 C2640,8 2760,72 2880,40 V80 H0 Z'
)


def waves(theme: str = "soft") -> None:
    """Animated wave divider between sections (cartoon-ocean transition)."""
    fills = {
        "soft": f"rgba(34,211,238,{0.10})",
        "deep": f"rgba(14,165,233,{0.16})",
        "faint": f"rgba(129,140,248,{0.10})",
    }
    fill = fills.get(theme, fills["soft"])
    svg = (
        f'<svg viewBox="0 0 2880 80" preserveAspectRatio="none" '
        f'stroke="{fill}" fill="{fill}"><path d="{_LAKE}"/></svg>'
    )
    st.markdown(f'<div class="oe-waves">{svg}{svg}</div>', unsafe_allow_html=True)


def section(title: str, meta: str = "") -> None:
    st.markdown(
        f'<div class="oe-sec"><span class="tick"></span><h3>{title}</h3>'
        f'<span class="meta">{meta}</span></div>',
        unsafe_allow_html=True,
    )


def glass_panel(html: str, *, animation: bool = True) -> None:
    st.markdown(f'<div class="oe-panel">{html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar
def source_pill(status: str, label: str, tooltip: str = "") -> str:
    """Status pill for one data source.

    ``live``      -> green pulsing "Live"            (fetched from the feed this session)
    ``cached``    -> neutral blue "Snapshot"          (verified offline snapshot; refresh available)
    ``downloaded``-> neutral blue "Snapshot"          (committed real download, no live fetch)
    ``sampled``   -> amber "Snapshot · demo grid"     (clearly-labelled committed demo grid)
    ``failed``    -> red "Unavailable"                (artifacts actually missing)
    anything else -> muted "Not integrated" (with the reason as tooltip)
    """
    if status == "live":
        cls, name = "live", "Live"
    elif status in ("cached", "downloaded"):
        cls, name = "cached", "Snapshot"
    elif status == "sampled":
        cls, name = "sample", "Snapshot · demo grid"
    elif status == "failed":
        cls, name = "fail", "Unavailable"
    else:
        cls, name = "off", "Not integrated"
    tip = f' title="{_html_escape(tooltip)}"' if tooltip else ""
    return (
        f'<span class="oe-pill {cls}"{tip}><span class="dot"></span>{label} — {name}</span>'
    )


def source_card(label: str, status_pill: str, tooltip: str = "") -> None:
    tip = f' title="{_html_escape(tooltip)}"' if tooltip else ""
    st.markdown(
        f'<div class="oe-source"{tip}><div class="row"><b style="font-size:.82rem">{label}</b>'
        f'</div>{status_pill}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- prediction
_TEMP_ANCHORS = [
    (0.0, (87, 92, 255)),     # deep blue  — very cold
    (10.0, (64, 156, 245)),   # blue        — cool
    (20.0, (45, 212, 191)),   # teal        — mild
    (27.0, (251, 191, 36)),   # amber       — warm
    (32.0, (248, 113, 113)),  # red         — hot
    (40.0, (220, 60, 60)),
]


def temp_color(t: float) -> str:
    """Continuous blue -> red gradient for a temperature in °C."""
    t = float(t)
    lo = _TEMP_ANCHORS[0][0]
    hi = _TEMP_ANCHORS[-1][0]
    if t <= lo:
        r, g, b = _TEMP_ANCHORS[0][1]
    elif t >= hi:
        r, g, b = _TEMP_ANCHORS[-1][1]
    else:
        for (t0, c0), (t1, c1) in zip(_TEMP_ANCHORS[:-1], _TEMP_ANCHORS[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-9)
                r, g, b = (c0[k] + (c1[k] - c0[k]) * f for k in range(3))
                break
    return f"rgb({int(r)},{int(g)},{int(b)})"


def depth_chips(levels: list[int], temps: np.ndarray) -> str:
    chips = []
    for level, t in zip(levels, temps):
        t = float(t)
        color = temp_color(t)
        chips.append(f'<div class="oe-chip"><b style="color:{color}">{t:.1f}°</b><span>{level} m</span></div>')
    return f'<div class="oe-chips">{"".join(chips)}</div>'


_FEATURE_LABELS = {"lat": "Lat (°)", "lon": "Lon (°)", "month": "Month", "sst": "SST (°C)",
                   "ssh": "SSH (m)", "sss": "SSS (psu)"}


def feature_bars(used: dict[str, float], ref_range: tuple[float, float] | None = None) -> str:
    vals = list(used.values())
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    bars = []
    for k, v in used.items():
        w = 100.0 * max(0.0, min(1.0, (v - lo) / (hi - lo)))
        label = _FEATURE_LABELS.get(k, k)
        val_txt = (str(int(v)) if k == "month" else f"{v:.2f}")
        bars.append(
            f'<div class="oe-bar"><label>{label}</label>'
            f'<div class="track"><div class="fill" style="width:{w:.0f}%"></div></div>'
            f'<span class="val">{val_txt}</span></div>'
        )
    return f'<div class="oe-barlist">{"".join(bars)}</div>'


def prediction_header(coord: str, status: str) -> None:
    st.markdown(
        f'<div class="oe-pred"><div class="glowline"></div>'
        f'<div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap">'
        f'<span class="oe-pulse-dot"></span>'
        f'<h3 style="margin:0;font-family:Outfit;font-weight:700;font-size:1.05rem">'
        f'Subsurface profile — {coord}</h3>'
        f'<span style="margin-left:auto;color:#8FB1CF;font-size:.78rem">{status}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def footer_html(fallback: bool = False) -> None:
    st.markdown(
        '<div class="oe-foot">'
        '<b>OceanEmbed</b> · SIH26066 · Space Technology<br>'
        'Trained on real Argo float profiles collocated to satellite surface fields'
        + (" · verified offline snapshot in use · live refresh available" if fallback else
           " · live satellite + Argo feeds connected")
        + '</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- charts
def profile_figure_plotly(
    actual: np.ndarray | None,
    predicted: np.ndarray,
    depths: np.ndarray,
    title: str = "",
    *,
    band: np.ndarray | None = None,
    baselines: list[dict[str, Any]] | None = None,
    emphasize: str | None = None,
    height: int = 500,
) -> Any:
    """Predicted-vs-actual temperature/depth chart.

    - Depth uses **uniform spacing** (category axis) so the thermocline isn't visually
      muted by non-linear metres; every standard level gets a gridline.
    - ``band`` (optional per-depth +/- sigma from nearby real Argo profiles) is drawn as
      a shaded region behind the predicted curve.
    - ``baselines`` (optional) are comparison profiles, each a dict with ``name``,
      ``values``, ``color`` and optional ``dash``; the one matching ``emphasize`` is
      drawn thicker/full-opacity, the rest thinner/faded.
    - Legend sits below the title on its own line (no overlap).
    """
    import plotly.graph_objects as go

    band = None if band is None else np.asarray(band, dtype=float)
    y = np.asarray(depths, dtype=float)
    baselines = [b for b in (baselines or []) if b.get("values") is not None]

    fig = go.Figure()

    # uncertainty band (behind everything)
    if band is not None and len(band) == len(predicted):
        fig.add_trace(go.Scatter(
            x=predicted + band, y=y, mode="lines", line=dict(width=0), hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=predicted - band, y=y, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(34,211,238,0.14)", name="±1σ nearby Argo spread",
            legendrank=3, hovertemplate="%{x:.2f} °C · %{y} m<extra>±1σ band</extra>",
        ))

    # real Argo profile (dashed)
    if actual is not None:
        fig.add_trace(
            go.Scatter(
                x=actual, y=y, name="Argo · real profile",
                mode="lines+markers", legendrank=2,
                line=dict(color=WARM, width=2, dash="dash"),
                marker=dict(size=9, color=WARM, symbol="diamond",
                            line=dict(color="#0a1a2f", width=1)),
                hovertemplate="%{x:.2f} °C · %{y} m<extra>Argo · real</extra>",
            )
        )

    # MLP prediction (solid)
    fig.add_trace(
        go.Scatter(
            x=predicted, y=y, name="OceanEmbed · MLP",
            mode="lines+markers", legendrank=1,
            line=dict(color=CYAN, width=3),
            marker=dict(size=11, color=CYAN, line=dict(color="#ffffff", width=1.5)),
            hovertemplate="%{x:.2f} °C · %{y} m<extra>OceanEmbed · MLP</extra>",
        )
    )

    # comparison baselines (climatology / linear regression …)
    for b in baselines:
        name = str(b["name"])
        emph = (emphasize == name)
        fig.add_trace(
            go.Scatter(
                x=np.asarray(b["values"], dtype=float), y=y, name=name,
                mode="lines", legendrank=4 if not emph else 0,
                line=dict(color=b["color"], width=(4.0 if emph else 2.2),
                          dash=str(b.get("dash", "solid"))),
                opacity=(1.0 if emph else 0.6),
                hovertemplate=f"%{{x:.2f}} °C · %{{y}} m<extra>{name}</extra>",
            )
        )

    z = list(zip(y, predicted))
    if actual is not None:
        z += [(d, a) for d, a in zip(y, actual) if a is not None]
    if band is not None and len(band) == len(predicted):
        z += [(d, p + b) for d, p, b in zip(y, predicted, band)]
        z += [(d, p - b) for d, p, b in zip(y, predicted, band)]
    for b in baselines:
        z += [(d, v) for d, v in zip(y, np.asarray(b["values"], dtype=float)) if np.isfinite(v)]
    if z:
        lo = min(x for _, x in z); hi = max(x for _, x in z)
        pad = max((hi - lo) * 0.25, 0.5)
        fig.update_xaxes(range=[lo - pad, hi + pad])

    fig.update_yaxes(
        type="category", categoryarray=[str(int(d)) for d in depths],
        categoryorder="array", autorange="reversed", nticks=7,
    )
    fig.update_layout(
        title=dict(text=title or "", font=dict(size=15, family="Outfit", color=TEXT),
                   x=0.01, y=0.985, xanchor="left", yanchor="top"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=24, r=18, t=98, b=24),
        font=dict(family="Inter", color=TEXT_MUTED, size=12),
        legend=dict(orientation="h", xanchor="left", x=0.01, y=1.045,
                    font=dict(size=11, color=TEXT_MUTED),
                    bgcolor="rgba(0,0,0,0)", itemwidth=38),
        hovermode="closest",
        transition=dict(duration=900, easing="cubic-in-out"),
    )
    fig.update_xaxes(
        gridcolor="rgba(103,232,249,0.10)", zeroline=False,
        title=dict(text="Temperature (°C)", font=dict(color=TEXT_MUTED)),
    )
    fig.update_yaxes(
        gridcolor="rgba(103,232,249,0.10)", zeroline=False,
        title=dict(text="Depth (m · uniform bins)", font=dict(color=TEXT_MUTED)),
    )
    return fig


def replay_profile_figure(
    months: list,
    profiles: np.ndarray,
    depths: np.ndarray,
    idx: int,
) -> Any:
    """Single-month predicted profile used by the time-series replay.

    ``months`` is a sequence of `pd.Timestamp`; the month at ``idx`` is drawn as the
    MLP prediction so the slider steps month-by-month.
    """
    import plotly.graph_objects as go

    y = np.asarray(depths, dtype=float)
    month = pd.Timestamp(months[idx])
    temps = np.asarray(profiles[idx], dtype=float)
    label = month.strftime("%b %Y")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=temps, y=y, name="OceanEmbed · MLP",
            mode="lines+markers", legendrank=1,
            line=dict(color=CYAN, width=3),
            marker=dict(size=11, color=CYAN, line=dict(color="#ffffff", width=1.5)),
            hovertemplate="%{x:.2f} °C · %{y} m<extra>MLP · " + label + "</extra>",
        )
    )
    lo, hi = float(temps.min()), float(temps.max())
    pad = max((hi - lo) * 0.25, 0.5)
    fig.update_xaxes(range=[lo - pad, hi + pad])

    fig.update_yaxes(
        type="category", categoryarray=[str(int(d)) for d in depths],
        categoryorder="array", autorange="reversed", nticks=7,
    )
    fig.update_layout(
        title=dict(text=f"Predicted 0–1000 m profile — {label}",
                   font=dict(size=15, family="Outfit", color=TEXT), x=0.01, y=0.985,
                   xanchor="left", yanchor="top"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=24, r=18, t=60, b=24),
        font=dict(family="Inter", color=TEXT_MUTED, size=12),
        hovermode="closest",
        transition=dict(duration=600, easing="cubic-in-out"),
    )
    fig.update_xaxes(
        gridcolor="rgba(103,232,249,0.10)", zeroline=False,
        title=dict(text="Temperature (°C)", font=dict(color=TEXT_MUTED)),
    )
    fig.update_yaxes(
        gridcolor="rgba(103,232,249,0.10)", zeroline=False,
        title=dict(text="Depth (m · uniform bins)", font=dict(color=TEXT_MUTED)),
    )
    return fig


def thermocline_figure(months: list, thermo_depths: list[float], title: str = "") -> Any:
    """Thermocline depth (steepest temperature drop) across the replayed months."""
    import plotly.graph_objects as go

    xs = [pd.Timestamp(m) for m in months]
    ys = list(thermo_depths)
    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys, mode="lines+markers", name="Thermocline depth",
            line=dict(color=VIOLET, width=2.5),
            marker=dict(size=8, color=VIOLET, line=dict(color="#ffffff", width=1)),
            hovertemplate="%{x|%b %Y} · thermocline %{y:.0f} m<extra>Thermocline</extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", nticks=6,
                     gridcolor="rgba(103,232,249,0.10)", zeroline=False,
                     title=dict(text="Thermocline depth (m)", font=dict(color=TEXT_MUTED)))
    fig.update_xaxes(gridcolor="rgba(103,232,249,0.10)", zeroline=False,
                     title=dict(text="Month", font=dict(color=TEXT_MUTED)))
    fig.update_layout(
        title=dict(text=title or "Thermocline depth over time",
                   font=dict(size=14, family="Outfit", color=TEXT), x=0.01, y=0.97,
                   xanchor="left", yanchor="top"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=24, r=18, t=48, b=24),
        font=dict(family="Inter", color=TEXT_MUTED, size=12),
        hovermode="closest",
    )
    return fig


def map_frame(html: str) -> None:
    """Shell that gives the folium map a rounded, glowing glass frame."""
    st.markdown(
        f'<div class="oe-map-frame" style="animation:cardRise .8s both">{html}</div>',
        unsafe_allow_html=True,
    )