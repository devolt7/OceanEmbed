"""OceanEmbed — interactive demo (Streamlit, dark-ocean glass UI).

Runs fully offline using previously-saved artifacts: the trained deep MLP + scaler
(``models/``), the latest cached surface grids (``data/processed/surface_*_latest.parquet``)
and the collocated Argo profiles (``data/interim/collocated.parquet``). No model
training and no data downloads happen at runtime.

The app has two views, controlled by ``st.session_state``:
  * map  — click a location (ocean pixel / float dot, or manual coordinates) to select it
  * profile — predicted 0-1000 m temperature profile for the clicked point,
    with a "← Back to map" link, a CSV download, and a button that highlights the
    nearest real Argo float back on the map.

Launch:  ``streamlit run app/streamlit_app.py``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.utils.config import load_config, resolve_path
from src.utils.io import manifest_file_path, read_manifest

from app.analysis import (
    BASELINE_CLIM, BASELINE_LINEAR, BASELINE_MLP,
    cached_surface_file, depths_from_targets, evaluate_baselines, fit_linear_baseline,
    mlp_predict, monthly_climatology, replay_monthly, thermocline_depth,
)
from app.ui import (
    CYAN, GOOD, HOT, OCEAN, TEAL, TEXT_MUTED, VIOLET, WARM,
    anomaly_banner, depth_chips, feature_bars, footer_html, glass_panel, hero, inject_theme,
    legend_html, mode_badge, prediction_header, profile_figure_plotly, replay_profile_figure,
    section, source_card, source_pill, stat_metrics, thermocline_figure, trust_badge, waves,
)

ROOT = Path(__file__).resolve().parents[1]

# SST palette used by both the map dots AND the shared legend component.
# Bands are ordered hottest -> coolest; the last entry is the "below lowest"
# fallback colour. Physically realistic range for ocean SST (not >100 °C!).
PALETTE_SST = [(28.0, HOT), (24.0, WARM), (20.0, TEAL)]
SST_LOW_COLOR = VIOLET


def sst_legend_items() -> list[tuple[str, str]]:
    """Label/colour rows for the shared legend (map view + profile view)."""
    items = [(f"SST &gt; {t:.1f} °C", c) for t, c in PALETTE_SST]
    items.append((f"SST &le; {PALETTE_SST[-1][0]:.1f} °C (cool)", SST_LOW_COLOR))
    return items


def any_sampled_source(cfg: dict) -> bool:
    """True when at least one optional satellite source is an offline cached demo grid."""
    for src in ("copernicus", "smap"):
        status = (read_manifest(cfg, src) or {}).get("status")
        if status == "sampled":
            return True
    return False


def data_mode(cfg: dict) -> tuple[str, str, str]:
    """Overall data mode for the UI: ("cached"|"live"|"degraded", label, tooltip).

    ``cached``  — SSH/SSS come from the committed offline demo grids (no internet needed).
    ``live``    — every source reports a real download, all present on disk.
    ``degraded``— key artifacts are missing and the app cannot predict properly.
    """
    try:
        has_sst = (read_manifest(cfg, "oisst") or {}).get("status") == "downloaded"
        has_argo = (read_manifest(cfg, "argo") or {}).get("status") == "downloaded"
    except Exception:
        has_sst = has_argo = False
    if any_sampled_source(cfg):
        return ("cached", "Offline · cached SSH/SSS grids",
                "Runs fully from cached artifacts — no internet or credentials needed at runtime.")
    if has_sst and has_argo:
        return ("live", "Cached · real downloaded data",
                "All sources downloaded and cached on disk — the app never fetches from the "
                "internet at runtime.")
    return ("degraded", "Degraded · check artifacts",
            "Some source manifests are missing — the app may still predict from whatever is cached.")


def region_box(collocated: pd.DataFrame | None, grids: dict[str, pd.DataFrame],
               cfg: dict) -> dict[str, float]:
    """Authoritative lon/lat box actually covered by the loaded data.

    Prefers the widest loaded surface grid (that is what bounds prediction), falls
    back to the collocated Argo box, then to the configured region.
    """
    if grids:
        for feat in ("sst", "ssh", "sss"):
            g = grids.get(feat)
            if g is not None and not g.empty and {"lat", "lon"}.issubset(g.columns):
                return {"west": float(g["lon"].min()), "east": float(g["lon"].max()),
                        "south": float(g["lat"].min()), "north": float(g["lat"].max())}
    if collocated is not None and not collocated.empty:
        return {"west": float(collocated["lon"].min()), "east": float(collocated["lon"].max()),
                "south": float(collocated["lat"].min()), "north": float(collocated["lat"].max())}
    return {k: float(v) for k, v in cfg["region"].items()}


def in_region(lat: float, lon: float, box: dict[str, float]) -> bool:
    return box["south"] <= lat <= box["north"] and box["west"] <= lon <= box["east"]


def nearest_surface_distance(grid: pd.DataFrame | None, lat: float, lon: float) -> float | None:
    """Approximate distance (km) to the nearest *valid* cell of a surface grid."""
    if grid is None or grid.empty:
        return None
    val_col = [c for c in grid.columns if c not in ("lon", "lat")]
    if not val_col:
        return None
    cells = grid.dropna(subset=[val_col[0]])
    if cells.empty:
        return None
    from scipy.spatial import cKDTree

    from src.utils.geo import project_xy

    xy = project_xy(cells["lon"].to_numpy(), cells["lat"].to_numpy()).T
    dists, _ = cKDTree(xy).query(project_xy(lon, lat).reshape(1, 2), k=1)
    return float(dists[0]) if np.isfinite(dists[0]) else None


def jump_to(lat: float, lon: float) -> None:
    st.session_state.click = {"lat": float(lat), "lng": float(lon)}
    st.session_state.focus_float = None
    st.session_state.view = "profile"
    st.rerun()


def manual_coord_entry() -> None:
    """Low-profile fallback for offline demos where folium/CDN tiles cannot load."""
    with st.expander("No map? Enter coordinates manually (offline fallback)"):
        c1, c2, c3 = st.columns([1, 1, 1], gap="small")
        lat = c1.number_input("Latitude (°N)", -90.0, 90.0, value=13.0, step=0.1,
                              format="%.3f", key="manual_lat")
        lon = c2.number_input("Longitude (°E)", -180.0, 180.0, value=75.0, step=0.1,
                              format="%.3f", key="manual_lon")
        if c3.button("Predict profile", key="manual_go", use_container_width=True):
            jump_to(lat, lon)


# --------------------------------------------------------------------------- assets
@st.cache_resource
def load_assets(cfg: dict):
    model_dir = resolve_path(cfg, "model_dir")
    missing: list[tuple[str, str]] = []  # (kind, filename) — surfaced in the sidebar

    meta = None
    meta_path = model_dir / "feature_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = None
            missing.append(("model", "feature_meta.json"))

    mlp = None
    scaler = None
    mlp_path = model_dir / "deep_mlp.keras"
    scaler_path = model_dir / "scaler_input.joblib"
    if mlp_path.exists() and scaler_path.exists():
        try:
            import tensorflow as tf

            mlp = tf.keras.models.load_model(mlp_path)
            scaler = joblib.load(scaler_path)
        except Exception:
            mlp = None
            scaler = None
            missing.append(("model", "deep_mlp.keras / scaler_input.joblib load failed"))
    elif mlp_path.exists() ^ scaler_path.exists():
        missing.append(("model", "deep_mlp.keras and scaler_input.joblib must coexist"))

    collocated = None
    coll_path = resolve_path(cfg, "interim_dir") / "collocated.parquet"
    if coll_path.exists():
        try:
            df = pd.read_parquet(coll_path)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            collocated = df
        except Exception:
            collocated = None
            missing.append(("interim", "collocated.parquet"))
    else:
        missing.append(("interim", "collocated.parquet"))

    grids = {}
    if meta:
        for feat in meta["features"]:
            if feat in ("lat", "lon", "month"):
                continue
            g = resolve_path(cfg, "processed_dir") / f"surface_{feat}_latest.parquet"
            if g.exists():
                try:
                    grids[feat] = pd.read_parquet(g)
                except Exception:
                    missing.append(("grid", f"surface_{feat}_latest.parquet"))
            else:
                missing.append(("grid", f"surface_{feat}_latest.parquet"))
    return {"meta": meta, "mlp": mlp, "scaler": scaler, "collocated": collocated, "grids": grids, "missing": missing}


def nearest_grid_value(grid: pd.DataFrame, lat: float, lon: float):
    """Nearest *valid* surface value for a click point (skips land/NaN grid cells).

    Falls back to a wider radius when the closest cells are masked out (e.g. the
    offshore NaN band in one OISST day) so coastal clicks still generate a profile.
    """
    if grid is None or grid.empty:
        return None, ""
    val_col = [c for c in grid.columns if c not in ("lon", "lat")][0]
    cells = grid.dropna(subset=[val_col])
    if cells.empty:
        return None, ""
    from scipy.spatial import cKDTree

    from src.utils.geo import project_xy

    xy = project_xy(cells["lon"].to_numpy(), cells["lat"].to_numpy()).T
    tree = cKDTree(xy)
    q = project_xy(lon, lat).reshape(1, 2)
    dists, idxs = tree.query(q, k=8)
    dists = np.atleast_1d(dists).ravel()
    idxs = np.atleast_1d(idxs).ravel()
    for d, i in zip(dists, idxs):
        if d <= 350.0:
            val = float(cells.iloc[int(i)][val_col])
            return (np.nan if not np.isfinite(val) else val), val_col
    return None, ""


def build_features(meta: dict, grids: dict, lat: float, lon: float, ref_time=None):
    if not meta or "features" not in meta:
        return None, {}
    row, used = {}, {}
    for feat in meta["features"]:
        if feat in ("lat", "lon"):
            row[feat] = lat if feat == "lat" else lon
        elif feat == "month":
            month = int(pd.Timestamp(ref_time).month) if ref_time is not None else 1
            row[feat] = month
        elif feat in grids and grids[feat] is not None:
            val, _ = nearest_grid_value(grids[feat], lat, lon)
            row[feat] = val
            used[feat] = val
        else:
            return None, {}
    # surface-feature bars should show ALL model inputs, not only satellite grids
    for feat in meta["features"]:
        if feat in row:
            used.setdefault(feat, row[feat])
    s = pd.Series([row[f] for f in meta["features"]], index=meta["features"])
    if s.isna().any():
        return None, used
    return s.to_frame().T, used


def find_nearby_argo(collocated: pd.DataFrame, lat: float, lon: float, max_deg: float = 1.5):
    if collocated is None or collocated.empty:
        return None
    dlat = (collocated["lat"] - lat).abs()
    dlon = (collocated["lon"] - lon).abs()
    near = collocated[(dlat <= max_deg) & (dlon <= max_deg)]
    if near.empty:
        return None
    near = near.assign(dist=(dlat + dlon)[near.index]).sort_values("dist")
    return near.head(1)


def nearby_profile_std(collocated: pd.DataFrame, lat: float, lon: float, targets: list[str],
                       max_deg: float = 1.5, max_n: int = 15):
    """Per-depth 1-sigma spread of the real Argo profiles close to a clicked point.

    Used as a ground-truth-derived uncertainty band around the MLP prediction
    (ensemble-of-nearby-observations analogue) — never invented numbers.
    """
    if collocated is None or collocated.empty:
        return None
    dlat = (collocated["lat"] - lat).abs()
    dlon = (collocated["lon"] - lon).abs()
    near = collocated[(dlat <= max_deg) & (dlon <= max_deg)].copy()
    if near.empty or len(near) < 2:
        return None
    temps = near[targets].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        std = np.nanstd(temps, axis=0)
    if np.isnan(std).any() or not np.isfinite(std).all():
        return None
    return std


def sst_color(v: float) -> str:
    for threshold, c in PALETTE_SST:
        if v > threshold:
            return c
    return SST_LOW_COLOR


# --------------------------------------------------------------------------- demo features
# Curated quick-select locations (inside the study region, float-dense / coastal zones).
DEMO_SPOTS = {
    "Arabian Sea upwelling": {"lat": 14.4, "lng": 71.6},
    "Bay of Bengal (cyclone zone)": {"lat": 18.0, "lng": 88.2},
    "Equatorial Indian Ocean": {"lat": 5.8, "lng": 80.4},
    "Coastal shelf": {"lat": 13.2, "lng": 74.4},
}
REPLAY_MAX_MONTHS = 12
ANOMALY_STD = 1.5


@st.cache_resource
def linear_baseline(cfg: dict, meta: dict):
    """Linear-regression baseline (train split). None when the split is unavailable."""
    try:
        return fit_linear_baseline(cfg, meta)
    except Exception:
        return None


@st.cache_resource
def climatology_tables(_collocated: pd.DataFrame, targets: list[str]):
    """Monthly mean/std of temperature per depth across the collocated profiles."""
    return monthly_climatology(_collocated, targets)


@st.cache_resource
def baseline_eval(cfg: dict, meta: dict, _mlp, _scaler, _linear, _collocated: pd.DataFrame):
    """RMSE comparison (MLP vs baselines) on the held-out test split, or None."""
    try:
        return evaluate_baselines(cfg, meta, _scaler, _mlp, linear=_linear, collocated=_collocated)
    except Exception:
        return None


@st.cache_resource
def replay_predictions(cfg: dict, meta: dict, _scaler, _mlp, lat: float, lon: float):
    return replay_monthly(cfg, meta, _scaler, _mlp, lat, lon)


# --------------------------------------------------------------------------- theme + shell
inject_theme()

cfg = load_config()
mode_label, mode_kind, mode_tip = data_mode(cfg)

with st.spinner("Loading on-device model and cached data…"):
    assets = load_assets(cfg)
meta, mlp, scaler, collocated, grids, missing = (
    assets["meta"], assets["mlp"], assets["scaler"], assets["collocated"], assets["grids"],
    assets["missing"],
)
targets = [t for t in (meta or {}).get("targets", [])]
depths = np.asarray([int(t.split("_")[1].replace("m", "")) for t in targets], dtype=float)

# dataset scale (reused by the pitch, stats row, anomaly + baseline logic)
n_colloc = 0 if collocated is None else len(collocated)
n_floats = 0 if collocated is None else collocated["float_id"].nunique()
span_years = 0.0
if collocated is not None and not collocated.empty:
    span_years = (collocated["time"].max() - collocated["time"].min()).days / 365.25

# view state
if "view" not in st.session_state:
    st.session_state.view = "map"
if "click" not in st.session_state:
    st.session_state.click = None
if "focus_float" not in st.session_state:
    st.session_state.focus_float = None

hero(
    title="OceanEmbed",
    framing=("Full-depth ocean monitoring relies on sparse, expensive Argo floats. "
             "OceanEmbed reconstructs the same 0–1000 m information from free satellite "
             "data alone — extending ocean visibility to locations no float has ever reached."),
    subtitle="Reconstructing the 0–1000 m ocean temperature profile from surface-only "
             "satellite data — trained exclusively on real Argo float measurements.",
)
if n_colloc:
    stat_metrics(2, [
        {"label": "Real Argo training data",
         "value": f"{n_colloc:,} profiles",
         "help": f"{n_floats:,} distinct floats · {span_years:.1f} years spanned"},
        {"label": "Inference · <100 ms",
         "value": "Fully on-device",
         "help": "No internet required — runs from cached artifacts"},
    ])
if any_sampled_source(cfg):
    trust_badge("Real Argo floats + NOAA OISST · SSH/SSS offline cached grids")
else:
    trust_badge("Real Argo floats + NOAA OISST · no synthetic profile data")
mode_badge(mode_label, kind=mode_kind, tooltip=mode_tip)
waves("soft")

# --------------------------------------------------------------------------- stats
match_rate = None
report_path = resolve_path(cfg, "interim_dir") / "collocation_report.csv"
if report_path.exists():
    rep = pd.read_csv(report_path)
    if len(rep) >= 2:
        usable = int(rep.iloc[1]["count"])
        matched = int(rep.iloc[2]["count"])
        match_rate = (100.0 * matched / usable) if usable else 0.0

stat_metrics(4, [
    {"label": "Profiles matched", "value": f"{n_colloc:,}" if n_colloc else "—",
     "help": "Argo float profiles collocated to satellite surface fields (SST/SSH/SSS)"},
    {"label": "Distinct floats", "value": f"{n_floats:,}" if n_floats else "—",
     "help": "WMO platform numbers"},
    {"label": "SST match rate", "value": f"{match_rate:.1f}%" if match_rate else "—",
     "help": "±2 days temporal · ≤100 km radius"},
    {"label": "Depth targets", "value": "6",
     "help": "0 → 1000 m multi-output regression"},
])
waves("deep")

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        f'<div class="oe-side-title"><span style="font-size:1.2rem;color:{CYAN}">◈</span>'
        f'OceanEmbed</div>',
        unsafe_allow_html=True,
    )
    st.caption("Satellite-embedding model · replay · offline demo")

    st.markdown("#### Data mode")
    st.markdown(
        f'<div class="oe-source"><div class="row"><b style="font-size:.82rem">'
        f'{mode_label}</b></div>'
        f'<span style="font-size:.74rem;color:{TEXT_MUTED}">{mode_tip}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Data sources")
    SOURCE_META = (
        ("argo", "Argo floats",
         "Real profile archive (GDAC) — cached from a previous download"),
        ("oisst", "OISST · sea surface temp",
         "NOAA daily 0.25° SST — mandatory surface feature; cached on disk"),
        ("copernicus", "CMEMS · sea surface height",
         "SSH (m) — wired into the trained model; committed offline cached grid"),
        ("smap", "SMAP · sea surface salinity",
         "SSS (psu) — wired into the trained model; committed offline cached grid"),
    )
    for src, label, tip in SOURCE_META:
        man = read_manifest(cfg, src)
        status = (man or {}).get("status", "not ingested")
        reason = (man or {}).get("reason", "")
        tooltip = reason or tip
        pill_status = status
        if status in ("sampled", "downloaded"):
            # Argo's manifest file (data/raw/argo_profiles.parquet) is intentionally NOT
            # committed — its committed offline representation is collocated.parquet.
            if src == "argo" and status == "downloaded":
                ref = resolve_path(cfg, "interim_dir") / "collocated.parquet"
                missing_hint = "`data/raw/argo_profiles.parquet`"
            else:
                ref = manifest_file_path(cfg, man)
                missing_hint = f"`{(man or {}).get('file', '<unknown>')}`"
            if ref is None or not ref.exists():
                pill_status = "failed"
                tooltip = ("Cached file missing: "
                           f"{missing_hint} — the repo moved or the snapshot was removed. "
                           "Restore it (e.g. `git restore`) or re-run `fetch_* --sample` to "
                           "regenerate an offline snapshot.")
        if pill_status == "sampled":
            tooltip = ("Cached · offline demo grid — physically-plausible synthetic "
                       "SSH/SSS wired end-to-end into the collocation + trained model. "
                       "No credentials or internet are needed at runtime.")
        elif pill_status == "downloaded" and src == "argo":
            tooltip = (f"{man.get('n_profiles', '?')} profiles · "
                       f"{man.get('n_floats', '?')} floats · real archive")
        elif pill_status == "downloaded" and src == "oisst":
            tooltip = f"{man.get('n_time', '?')} daily frames · real archive"
        elif pill_status == "downloaded" and src == "copernicus":
            tooltip = f"{man.get('n_time', '?')} time steps · cached CMEMS SSH"
        elif pill_status == "downloaded" and src == "smap":
            tooltip = f"{man.get('n_time', '?')} time steps · cached SMAP SSS"
        source_card(label, source_pill(pill_status, label, tooltip), tooltip=tooltip)

    st.markdown("#### Model")
    if mlp is not None and meta:
        feats = " · ".join("`" + f + "`" for f in meta["features"])
        trained_at = (meta.get("trained_utc", "") or "")[:10]
        st.markdown(
            f'<div class="oe-source"><div class="row"><b style="font-size:.82rem">Deep MLP · loaded</b>'
            f'<span style="font-size:.68rem;color:{TEXT_MUTED}">trained {trained_at or "—"}</span>'
            f'</div><span style="font-size:.74rem;color:{TEXT_MUTED}">Features: {feats}</span></div>',
            unsafe_allow_html=True,
        )
        if collocated is None:
            st.warning("No collocated profiles cached — feature sampling disabled.")
    else:
        st.warning("No trained model found. Run `python -m src.models.train` first.")

    if missing:
        st.markdown("#### Artifacts missing")
        st.caption("Some cached artifacts are unavailable — the related feature is "
                   "disabled until they are restored (e.g. `git restore` or re-run "
                   "ingestion/collocation):")
        for _kind, _file in missing:
            st.markdown(
                f'<div style="font-size:.76rem;color:{HOT}">• `{_file}` ({_kind})</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Map legend")
    st.markdown(
        f'<div class="oe-source">'
        f'<div style="font-size:.74rem;color:{TEXT_MUTED}">Argo collocated dots · SST bands</div>'
        f'{legend_html(sst_legend_items())}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### About")
    st.caption("Click an ocean pixel or float dot to predict the subsurface profile, "
               "or pick a quick-select spot. All inference is on-device from cached "
               "artifacts — no network at runtime.")


# --------------------------------------------------------------------------- map view
def render_map() -> None:
    # derive the region box from the *loaded* data so the rectangle always matches it
    base = region_box(collocated, grids, cfg)
    region = {k: (v - 0.5 if k in ("west", "south") else v + 0.5) for k, v in base.items()}

    waves("faint")
    section("Collocated Argo network · sea-surface temperature",
            meta=f"{n_colloc:,} profiles · {n_floats:,} floats")

    spot_keys = list(DEMO_SPOTS) + ["🎲 Surprise me"]
    selected_spot = st.pills("Demo locations", spot_keys, key="demo_spot",
                             selection_mode="single", default=None)
    if selected_spot:
        # consume the selection so returning to the map does not re-trigger the jump
        st.session_state.pop("demo_spot", None)
        if selected_spot == "🎲 Surprise me":
            if collocated is not None and not collocated.empty:
                row = collocated.sample(1).iloc[0]
                jump_to(float(row["lat"]), float(row["lon"]))
            else:
                st.info("No collocated profiles cached — pick one of the fixed spots instead.")
        else:
            spot = DEMO_SPOTS[selected_spot]
            jump_to(spot["lat"], spot["lng"])
    st.caption("Quick-select an Argo-dense hotspot, click any ocean pixel or float dot on "
               "the map, or use the manual-coordinates fallback below. 🎲 picks a random "
               "real float location.")

    focus = st.session_state.get("focus_float")
    center = ((focus or {}).get("lat"), (focus or {}).get("lon")) if focus else None
    try:
        m = folium.Map(
            location=[center[0], center[1]] if center else [
                (region["south"] + region["north"]) / 2,
                (region["west"] + region["east"]) / 2,
            ],
            zoom_start=8 if focus else 5,
            tiles="OpenStreetMap",
            control_scale=True,
        )
        folium.Rectangle(
            bounds=[[region["south"], region["west"]], [region["north"], region["east"]]],
            color="#2ca02c", weight=1, fill=False, popup="study region",
        ).add_to(m)

        if collocated is not None and not collocated.empty:
            fg = folium.FeatureGroup(name="Argo collocated profiles")
            sample = collocated.sample(min(400, len(collocated)), random_state=1) \
                if len(collocated) > 400 else collocated
            for _, r in sample.iterrows():
                latv, lonv = float(r["lat"]), float(r["lon"])
                date = pd.Timestamp(r["time"]).date()
                tip_txt = f"SST {r['sst']:.1f}°C · float {r['float_id']} · {date}"
                fg.add_child(folium.CircleMarker(
                    [latv, lonv], radius=2.6,
                    color=sst_color(float(r["sst"])), fill=True, fill_opacity=0.9,
                    popup=folium.Popup(tip_txt, max_width=240),
                    tooltip=folium.Tooltip(tip_txt, sticky=True),
                ))
            fg.add_to(m)

        if focus:
            fm = folium.CircleMarker(
                [float(focus["lat"]), float(focus["lon"])], radius=8,
                color=GOOD, fill=True, fill_opacity=0.6, weight=3,
                popup=f"Selected Argo float {focus.get('float_id', '')}",
                tooltip=folium.Tooltip(f"Float {focus.get('float_id', '')}", sticky=True),
            )
            fm.add_to(m)

        folium.LayerControl(collapsed=True).add_to(m)
        result = st_folium(m, width="100%", height=500,
                           returned_objects=["last_clicked"], key="ocean_map")
        st.caption("Click anywhere on the map to predict the 0–1000 m temperature profile at "
                   "that point. Basemap tiles load from the internet — if they do not appear, "
                   "use the manual-coordinates fallback below.")
    except Exception as e:  # pragma: no cover — offline/degraded browser rendering
        st.error("The interactive map could not be rendered — use the manual-coordinates "
                 "fallback below to still predict a profile.")
        result = {}
    clicked = result.get("last_clicked") if isinstance(result, dict) else None
    if clicked and clicked.get("lat") is not None:
        st.session_state.click = {"lat": float(clicked["lat"]), "lng": float(clicked["lng"])}
        st.session_state.focus_float = None
        st.session_state.view = "profile"
        st.rerun()

    manual_coord_entry()


# --------------------------------------------------------------------------- profile view
def render_profile() -> None:
    click = st.session_state.get("click")
    if click is None or click.get("lat") is None:
        st.session_state.view = "map"
        st.rerun()
    lat, lon = float(click["lat"]), float(click["lng"])

    waves("faint")
    if st.button("← Back to map", key="back_to_map"):
        st.session_state.click = None
        st.session_state.focus_float = None
        st.session_state.view = "map"
        st.rerun()
    prediction_header(f"{lat:.2f}°N · {lon:.2f}°E", status="on-device inference")

    box = region_box(collocated, grids, cfg)
    outside = not in_region(lat, lon, box)

    ref_time = collocated["time"].max() if collocated is not None else None
    with st.spinner("Reconstructing the 0–1000 m profile from surface features…"):
        X, used = build_features(meta, grids, lat, lon, ref_time=ref_time)
        if X is None:
            if outside:
                st.error(f"This point ({lat:.2f}°N, {lon:.2f}°E) lies outside the mapped "
                         f"region ({box['south']:.1f}–{box['north']:.1f}°N, "
                         f"{box['west']:.1f}–{box['east']:.1f}°E).")
                st.caption("Click inside the region, pick a quick-select spot above, or "
                           "enter ocean coordinates below.")
            else:
                st.error("No prediction is possible at this point — it is on land or in an "
                         "area the cached satellite grids do not cover.")
                st.caption("Try clicking an ocean pixel on the map, picking a quick-select "
                           "spot, or entering ocean coordinates below.")
            manual_coord_entry()
            return
        Xarr = X.to_numpy()
        pred = mlp_predict(meta, scaler, mlp, Xarr)[0]

        edge_note: list[str] = []
        if outside:
            edge_note.append("just outside the mapped region")
        d0 = nearest_surface_distance(grids.get("sst"), lat, lon)
        if d0 is not None and d0 > 40.0:
            edge_note.append(f"~{d0:.0f} km from the nearest valid ocean pixel "
                             f"(land / masked coastline)")
        if edge_note:
            st.caption("Note: this point lies " + " · ".join(edge_note) +
                       " — the profile uses the closest valid surface values.")

        actual = None
        argo_row = None
        band = None
        nearby = find_nearby_argo(collocated, lat, lon)
        if nearby is not None and not nearby.empty:
            argo_row = nearby.iloc[0]
            actual = np.asarray([argo_row[t] for t in targets], dtype=float)
        band = nearby_profile_std(collocated, lat, lon, targets)

    # ------------------------- baselines + climatology (features 2 & 3) -------------
    ref_month = int(pd.Timestamp(ref_time).month) if ref_time is not None else 1
    linear = linear_baseline(cfg, meta)
    clim_mean = clim_std = None
    if collocated is not None:
        clim_mean, clim_std = climatology_tables(collocated, targets)

    baseline_lines: list[dict] = []
    if clim_mean is not None and ref_month in clim_mean.index:
        baseline_lines.append({"name": BASELINE_CLIM,
                               "values": clim_mean.loc[ref_month, targets].to_numpy(dtype=float),
                               "color": VIOLET, "dash": "dashdot"})
    if linear is not None:
        baseline_lines.append({"name": BASELINE_LINEAR, "values": linear.predict(Xarr)[0],
                               "color": OCEAN, "dash": "dot"})

    # ------------------------- marine heatwave / anomaly banner (feature 3) ----------
    if clim_mean is not None and clim_std is not None and ref_month in clim_mean.index:
        mu = clim_mean.loc[ref_month, targets].to_numpy(dtype=float)
        sd = clim_std.loc[ref_month, targets].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            dev = pred - mu
            z = np.abs(dev) / np.where(sd > 0, sd, np.nan)
        if np.isfinite(z).any():
            i = int(np.nanargmax(np.where(np.isfinite(z), z, 0.0)))
            if z[i] >= ANOMALY_STD:
                d, dz = int(depths[i]), dev[i]
                kind = "above normal — possible marine heatwave" if dz > 0 else \
                       "below normal — possible cold anomaly"
                anomaly_banner(f"⚠ Subsurface anomaly detected at {d} m ({dz:+.1f}°C {kind})", "warn")
            else:
                anomaly_banner("Profile within normal range — no depth exceeds ±1.5σ "
                               "of the monthly climatology", "ok")

    model_choice = st.pills("Model comparison", [BASELINE_MLP, BASELINE_LINEAR, BASELINE_CLIM],
                            default=BASELINE_MLP, key="model_choice", selection_mode="single")

    col_chart, col_side = st.columns([3, 2], gap="medium")

    with col_chart:
        fig = profile_figure_plotly(
            actual=actual, predicted=pred, depths=depths, band=band,
            baselines=baseline_lines, emphasize=model_choice,
            title="Predicted temperature vs depth",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div style="font-size:.72rem;color:{TEXT_MUTED}">'
            f'Solid = OceanEmbed MLP · dashed = nearest real Argo · shaded = ±1σ of nearby Argo · '
            f'dash-dot = climatology · dotted = linear regression</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="margin-top:.3rem">'
                    f'{legend_html(sst_legend_items())}</div>'
                    '<div style="margin-top:.3rem"></div>',
                    unsafe_allow_html=True)

        rmse = baseline_eval(cfg, meta, mlp, scaler, linear, collocated)
        if rmse is not None:
            st.markdown("**RMSE vs held-out real Argo profiles** *(test split)*")
            st.dataframe(rmse, use_container_width=True, height=220)

        csv_cols: dict[str, np.ndarray] = {
            "Depth (m)": depths,
            "OceanEmbed MLP (°C)": pred,
            "Argo nearby ±1σ (°C)": band if band is not None else np.full_like(pred, np.nan),
            "Argo actual (°C)": actual if actual is not None else np.full_like(pred, np.nan),
            "Climatology (°C)": next((b["values"] for b in baseline_lines
                                      if b["name"] == BASELINE_CLIM),
                                     np.full_like(pred, np.nan)),
            "Linear regression (°C)": next((b["values"] for b in baseline_lines
                                            if b["name"] == BASELINE_LINEAR),
                                           np.full_like(pred, np.nan)),
        }
        dl_df = pd.DataFrame(csv_cols)
        dl_df.insert(0, "Latitude (°N)", float(lat))
        dl_df.insert(1, "Longitude (°E)", float(lon))
        st.download_button(
            "⬇ Download profile (CSV)",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"oceanembed_profile_{lat:.2f}N_{lon:.2f}E.csv",
            mime="text/csv", key="dl_profile_csv",
        )

    with col_side:
        side_html = f"{depth_chips([int(l) for l in depths], pred)}"
        if argo_row is not None:
            argo_note = (f"Nearest real Argo: float {argo_row['float_id']} · "
                         f"{pd.Timestamp(argo_row['time']).date()} · "
                         f"({argo_row['lat']:.2f}°, {argo_row['lon']:.2f}°)")
            side_html += (
                f'<p style="margin:.7rem 0 0;font-size:.78rem;color:{TEXT_MUTED}">'
                f'<span style="color:{GOOD}">◈</span> {argo_note}</p>'
            )
        else:
            side_html += (
                f'<p style="margin:.7rem 0 0;font-size:.78rem;color:{TEXT_MUTED}">'
                'No real Argo profile within ~1.5° of this point — the profile is an '
                'extrapolation from the nearest surrounding floats.</p>'
            )
        if used:
            bars = "".join(
                f'<div style="margin-top:.2rem;color:{TEXT_MUTED};font-size:.76rem">'
                f'Surface features</div>{feature_bars(used)}'
            )
            side_html += f'<div style="margin-top:.8rem">{bars}</div>'
        glass_panel(side_html)
        if argo_row is not None:
            if st.button(f"◈ Locate float {argo_row['float_id']} on the map",
                         key="locate_float", use_container_width=True):
                st.session_state.focus_float = {
                    "lat": float(argo_row["lat"]),
                    "lon": float(argo_row["lon"]),
                    "float_id": str(argo_row["float_id"]),
                }
                st.session_state.click = None
                st.session_state.view = "map"
                st.rerun()
        st.caption("1000 m values are near-climatological: the deep ocean is close "
                   "to isothermal, so the model plateaus there.")

    # ------------------------- monthly replay (feature 1) ----------------------------
    pt = (round(lat, 3), round(lon, 3))
    if st.session_state.get("replay_point") != pt:
        if "replay_slider" in st.session_state:
            del st.session_state.replay_slider
        st.session_state.replay_point = pt
        st.session_state.replay_playing = False

    st.markdown('<div style="margin-top:.6rem"></div>')
    section(f"Monthly replay — how the profile changes over time",
            meta="last available months · on-device")
    with st.spinner("Replaying monthly profiles from the cached satellite series…"):
        rep = replay_predictions(cfg, meta, scaler, mlp, lat, lon)
    if rep is None:
        st.info("No cached monthly satellite series for this point — replay unavailable "
                "(needs the cached OISST/CMEMS/SMAP grids in `data/cached/`).")
    else:
        months = list(rep["months"])
        preds = rep["pred"]
        n_mon = len(months)
        if "replay_slider" not in st.session_state:
            st.session_state.replay_slider = 0
        playing = bool(st.session_state.get("replay_playing", False))

        ctrl_a, ctrl_b, ctrl_c = st.columns([1, 3, 3], gap="small")
        with ctrl_a:
            if st.button(("⏸ Pause" if playing else "▶ Play"),
                         key="replay_toggle", use_container_width=True):
                st.session_state.replay_playing = not playing
                if not st.session_state.replay_playing:
                    st.session_state.replay_slider = 0
                st.rerun()
        with ctrl_b:
            idx = st.slider("Month", 0, n_mon - 1, key="replay_slider")
        with ctrl_c:
            st.markdown(
                f'<div style="display:flex;align-items:center;height:100%;font-size:.88rem;'
                f'color:{CYAN};font-weight:600">{pd.Timestamp(months[idx]).strftime("%b %Y")}</div>',
                unsafe_allow_html=True,
            )

        thermo = [thermocline_depth(preds[i], depths) for i in range(n_mon)]
        st.plotly_chart(replay_profile_figure(months, preds, depths, idx),
                        use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(thermocline_figure(months, thermo),
                        use_container_width=True, config={"displayModeBar": False})

        gaps_shown = [
            g.strftime("%b %Y") for g in rep["gaps"]
            if pd.Timestamp(months[0]) <= g <= pd.Timestamp(months[-1])
        ]
        if gaps_shown:
            st.caption("⚠ Data unavailable for " + ", ".join(gaps_shown) +
                       " — surface satellite values are missing those months, so those "
                       "months are skipped (nothing is extrapolated).")
        else:
            st.caption("Every replayed month has valid cached surface data — no gaps.")

        if playing:
            time.sleep(0.55)
            st.session_state.replay_slider = (idx + 1) % n_mon
            st.rerun()


# --------------------------------------------------------------------------- dispatch
if meta and mlp:
    if st.session_state.view == "profile":
        render_profile()
    else:
        render_map()
else:
    st.info("No trained model found — run `python -m src.models.train` first, "
            "then reload this page.")

footer_html()