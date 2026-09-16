# OceanEmbed

**Satellite embedding–based deep learning framework for reconstructing subsurface ocean
temperature (0–1000 m) from surface satellite observations alone.**

**Problem statement:** SIH26066 · **Theme:** Space Technology · **Category:** Software

Only ~4,000 Argo floats monitor subsurface ocean temperature worldwide; each float costs
~$20,000+ and deployment is slow. OceanEmbed instead learns to predict the **full
subsurface temperature profile from surface-only satellite data** (sea surface
temperature / height / salinity), trained and validated exclusively against **real Argo
float measurements — zero synthetic data**.

This MVP is scoped to the **Arabian Sea + Bay of Bengal** (0–25°N, 50–100°E) so the whole
pipeline runs on a free-tier GPU session (Google Colab) or a laptop.

---

## Architecture

```
                    ┌──────────── Argo float profiles ────────────┐
                    │ (argopy / Ifremer GDAC·ERDDAP, no auth)     │   data/raw/
                    └────────────────────┬────────────────────────┘
                                         │
 NOAA OISST daily SST ───┐               │
 (ERDDAP griddap, no auth)│               │
                         ▼               ▼
                    ┌──────────────────────────┐   KDTree nearest grid cell
                    │  COLLOCATION ENGINE      │   ± 2-day temporal window
                    │   · interpolate Argo T(z)│   depth interpolation → 6 std levels
   CMEMS SSH ──────►│     onto 6 target levels │   drop < 500 m profiles
   (optional, .env) │   · sample SST/SSH/SSS   │                 │
   SMAP SSS ───────►│     per profile          │                 ▼
   (optional, .env) └──────────────────────────┘        data/interim/collocated.parquet
                                                                   │
                                     train/val/test by ● FLOAT ID (no leakage)
                                                                   ▼
                                        ┌────────────────────────────────────┐
                                        │  BASELINES: RandomForest / XGBoost  │
                                        │  DEEP: Keras MLP (multi-output head)│
                                        │  (stretch: patch CNN-LSTM)          │
                                        └────────────────────────────────────┘
                                                                   │
                            reports/figures ──── evaluate.py ──── models/*.keras,
                            reports/results_summary.md            scaler, meta
                                                                   │
                                                    streamlit_app.py (offline demo map)
```

### Pipeline stages
| Phase | Script | Output |
|------|--------|--------|
| 1 Ingestion | `python -m src.ingestion.fetch_argo` · `fetch_oisst` · `fetch_copernicus` · `fetch_smap` · `fetch_all` | `data/raw/*.nc`, `*.parquet`, per-source `manifest_*.json` |
| 2 Collocation | `python -m src.collocation.collocate` | `data/interim/collocated.parquet`, match-rate report |
| 3–4 Models | `python -m src.models.train [--fast]` | `models/baseline_rf.joblib`, `baseline_xgb.joblib`, `deep_mlp.keras`, scaler, `feature_meta.json` |
| 5 Evaluation | `python -m src.evaluation.evaluate` | `reports/metrics_comparison.csv`, `profile_*.png`, `results_summary.md` |
| 6 Demo | `streamlit run app/streamlit_app.py` | interactive map (offline) |
| 7 Colab | `notebooks/oceanembed_colab.ipynb` | end-to-end GPU run |

---

## How to run the project

There are two ways: **Method 1** — run everything from a terminal with the project
folder as the working directory; **Method 2** — run it from an IDE (VS Code shown here).

Both methods need **Python 3.10** and the same first-time setup (virtual environment +
dependencies). Pick whichever fits your workflow.

---

### Method 1 — Terminal (inside the project folder)

1. **Open a terminal and enter the project folder**
   ```bash
   cd "/home/mrdevolt/CAREER/Hackathon Projects/OceanEmbed"
   ```

2. **Create and activate a virtual environment** (one-time setup)
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   # on Windows instead: .venv\Scripts\activate
   ```
   Your prompt should now show a `(.venv)` prefix.

3. **Install dependencies** (one-time setup)
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env             # fill in CMEMS / Earthdata credentials if you have them
   ```
   Without credentials the pipeline still runs in SST-only "degraded mode".
   You can also use `--sample` to generate clearly-labelled demo grids for CMEMS/SMAP so the full
   6-feature pipeline can run offline (see *Demo-sample CMEMS/SMAP* below).

5. **Run the full pipeline** (uses the short 13-month `config.bootstrap.yaml`)
   ```bash
   python -m src.ingestion.fetch_all --config config.bootstrap.yaml
   # or, to use demo-sample SSH/SSS grids (no credentials needed):
   # python -m src.ingestion.fetch_all --config config.bootstrap.yaml --sample
   python -m src.collocation.collocate --config config.bootstrap.yaml
   python -m src.models.train --config config.bootstrap.yaml
   python -m src.evaluation.evaluate --config config.bootstrap.yaml
   ```

6. **Run the interactive demo** (opens a browser with the map)
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Press `Ctrl+C` in the terminal to stop the demo.

7. **Run the tests** (optional)
   ```bash
   pytest
   ```

> **Tip:** each new terminal session must repeat step 2 (`source .venv/bin/activate`)
> before running the pipeline commands.

---

### Method 2 — VS Code

1. **Open the project in VS Code**
   - Open VS Code → `File` → `Open Folder…` → select the
     `OceanEmbed` project folder.
   - Or from a terminal: `code "/home/mrdevolt/CAREER/Hackathon Projects/OceanEmbed"`

2. **Select the Python interpreter / virtual environment**
   - Open any `.py` file → VS Code may prompt to select an interpreter.
   - Or press `Ctrl+Shift+P` → type `Python: Select Interpreter` → choose
     the one pointing at `OceanEmbed/.venv/bin/python` (the project's virtualenv).
   - This makes every terminal you open *inside* VS Code use the `.venv` automatically.

3. **Install dependencies** (one-time setup)
   - Open a VS Code terminal (`Terminal` → `New Terminal`) and run:
     ```bash
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

4. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   ```

5. **Run the pipeline step-by-step from a VS Code terminal**
   - `Terminal` → `New Terminal`, then run the four pipeline commands:
     ```bash
     python -m src.ingestion.fetch_all --config config.bootstrap.yaml
     python -m src.collocation.collocate --config config.bootstrap.yaml
     python -m src.models.train --config config.bootstrap.yaml
     python -m src.evaluation.evaluate --config config.bootstrap.yaml
     ```

6. **Run the interactive demo from VS Code**
   - In a VS Code terminal:
     ```bash
     streamlit run app/streamlit_app.py
     ```
     Streamlit prints a URL (usually `http://localhost:8501`) — click it or
     `Ctrl+Click` it to open the demo in your browser.
   - Stop it with `Ctrl+C` in the terminal.
   - *(Tip: you can also run the app by opening `app/streamlit_app.py` and clicking the
     ▶ **Run Python File** button if the Streamlit launcher is configured, but the
     terminal command above is the reliable way.)*

7. **Run the tests from VS Code**
   ```bash
   pytest
   ```
   Or use the built-in **Testing** panel (flask/test tube icon in the left sidebar)
   after selecting the interpreter in step 2 — VS Code auto-discovers the `tests/`
   folder via `pytest.ini`.

---

> The repository ships with a **full 5-year window** in `config.yaml` and a
> **short 13-month quick-start window** in `config.bootstrap.yaml`. The steps above use
> the short window so download volume and training time stay free-tier friendly.
> Raise the window/region in `config.yaml` (or *any* of those values) and re-run — the
> whole pipeline is config-driven.

### Google Colab
Open `notebooks/oceanembed_colab.ipynb` and upload the repo folder (or mount Drive),
then run top-to-bottom on a GPU runtime. The notebook installs dependencies, runs
ingestion → collocation → training → evaluation, and renders the comparison figures.

---

## Data sources — 100% real data

| Source | Data | Access | Credentials |
|--------|------|--------|-------------|
| **Argo** (targets) | float temperature/depth profiles | `argopy` (Ifremer GDAC/ERDDAP), Argovis HTTPS fallback | none |
| **NOAA OISST** (SST) | daily sea-surface temperature, 0.25° | public ERDDAP griddap subset | none |
| **Copernicus Marine** (SSH) | daily L4 sea-level anomaly | `copernicusmarine` toolbox | free CMEMS account → `.env` |
| **NASA SMAP** (SSS) | L3 sea-surface salinity 8-day composite | `earthaccess` (PO.DAAC) | free Earthdata login → `.env` |

**Degraded mode:** if CMEMS / Earthdata credentials are absent, those scripts *skip
gracefully* (a `manifest_*.json` records `status: skipped`) and their feature column is
simply **omitted** from the models. SST is mandatory — the pipeline refuses to run with
no satellite SST. Every fallback documented here still uses **real data**, just a smaller
feature set.

### Demo-sample CMEMS/SMAP (offline)

To exercise the full 6-feature pipeline offline before obtaining credentials, use:

```bash
python -m src.ingestion.fetch_all --config config.bootstrap.yaml --sample
python -m src.collocation.collocate --config config.bootstrap.yaml
python -m src.models.train --config config.bootstrap.yaml
```

The `--sample` flag writes a deterministic, physically-plausible **demo grid** for SSH/SSS
(`status: "sampled"` in the manifest). These grids are **fully wired end-to-end**: they are
collocated onto every Argo profile, become real columns (`ssh`/`sss`) in the training matrix,
and the model is retrained with all six features. The Streamlit sidebar shows them with a
green **"Cached · offline"** pill (not the live pill, and never passed off as a real
download). The trust badge ("Real Argo floats + NOAA OISST") still applies to the Argo
profiles and SST data; the demo-sample SSH/SSS fields are clearly labelled synthetic and
must be replaced with a live download before any real-world use. See
`src/ingestion/sample_data.py` for implementation details.

### Live integration (with credentials)

Once you have the required accounts, fill in `.env` and run the live pipeline:

```bash
# 1. Add real SSH/SSS data
python -m src.ingestion.fetch_copernicus --config config.bootstrap.yaml  # SSH
python -m src.ingestion.fetch_smap --config config.bootstrap.yaml        # SSS
# 2. Re-collocate (adds ssh/sss columns)
python -m src.collocation.collocate --config config.bootstrap.yaml
# 3. Retrain the model with the 6-feature input
python -m src.models.train --config config.bootstrap.yaml
# 4. Reload the Streamlit app — it auto-detects the new feature_meta.json
```

**Credentials required:**
- **CMEMS SSH:** Free account at https://data.marine.copernicus.eu/register →
  `COPERNICUSMARINE_USERNAME` + `COPERNICUSMARINE_PASSWORD` in `.env`.
- **SMAP SSS:** Free NASA Earthdata login at https://urs.earthdata.nasa.gov/ →
  `NASA_EARTHDATA_USERNAME` + `NASA_EARTHDATA_PASSWORD` in `.env`.

## Configuration (`config.yaml`)

Region box, time range, target depth levels, minimum Argo depth, collocation window
(±2 days) / match radius, feature list, train-val-test fractions, and all
hyper-parameters (RF / XGB / MLP) are config values — not hardcoded. `path.*` entries
point at the standard `data/…`, `models/…`, `reports/…` directories.

## Model design notes

* **Multi-output models.** Each model family predicts all six depth temperatures in one
  forward pass (RF via `MultiOutputRegressor`, XGBoost via native `multi_output_tree`,
  MLP via a shared 6-unit output head). One shared model is simpler and makes the deep
  model a fair apples-to-apples benchmark.
* **Leakage-free split by float ID.** Splitting randomly by row would leak because
  successive profiles of the same float are autocorrelated. OceanEmbed splits on the
  WMO platform number so a float's profiles never span train *and* test
  (`src/utils/datasets.py`, tested). The split indices are persisted to
  `data/processed/` and reused by evaluation.
* **Deep model.** Keras MLP `[128, 64]` (relu, dropout 0.05) → 6 outputs, Adam/MSE,
  early stopping on val loss, best checkpoint saved to `models/deep_mlp.keras`. The MLP
  predicts **centred targets** (Y minus its train-set mean, stored as `targets_mean` in
  `feature_meta.json` and re-added at inference) — this lets the net focus on the
  surface-driven residual and is why the MLP edges out a plain linear regression in the
  in-app RMSE comparison. Inputs are standardised with a fitted `StandardScaler`.
* **Stretch goal (implemented, off by default):** a CNN-LSTM variant that ingests a
  small space-time SST **patch** per point (`src/models/deep_model.py::build_cnn_lstm`);
  requires `cfg.model.mlp.use_cnn_lstm: true` and surface patch data, planned for the
  next milestone.

## Collocation details

* **Spatial:** nearest satellite grid cell via an equirectangular-projected `scipy` KDTree,
  capped at `max_distance_km` (default 100 km).
* **Temporal:** surface snapshot within ±`temporal_window_days` (default 2 d) of the float
  profile; SMAP 8-day composites use a wider tolerance (`ingestion.smap.window_days`).
* **Depth:** linear interpolation of each profile's T(z) onto 0/50/100/200/500/1000 m;
  profiles that don't reach 500 m are dropped. Target levels below the profile's deepest
  real sample become NaN (never invented); the surface level clamps to the shallowest
  real Argo observation when within 30 m (standard Argo practice).
* **Match reporting:** the collocation report logs how many raw profiles survived each
  stage and prints the final SST match rate (`interim/collocation_report.csv`).

## Results (auto-generated)

`python -m src.evaluation.evaluate` prints and saves a per-depth RMSE/MAE/R² comparison
table (`reports/metrics_comparison.csv`), plots predicted-vs-actual profiles for sample
test floats (`reports/figures/profile_*.png`), and writes a plain-language
`reports/results_summary.md`. **Run it yourself** — the numbers depend on your exact
window/region/credentials.

### Verified end-to-end smoke run (this repo, 13-month window)

Ran with the bundled `config.bootstrap.yaml` (lat 5–25, lon 55–95, 2024-09-01 → 2025-09-30,
full 6-feature pipeline — SSH/SSS from the clearly-labelled offline demo grids, no
credentials configured). Full pipeline, Argo + OISST from real data:

| Stage | Result |
|-------|--------|
| OISST | 394 daily frames (13 months), 0.25° |
| Argo | **3,540 profiles / 101 floats** (QC-filtered, 0–1200 m) |
| Collocation | **3,221** target profiles matched to SST, ±2 d / 100 km, **100% match rate** |
| Models | RF / XGBoost / Keras-MLP, all 6 depths multi-output, float-ID split |

Held-out test MAE by depth (°C):

| Depth | RF | XGB | MLP |
|-------|-----|-----|-----|
| 0 m | 0.22 | 0.25 | 0.21 |
| 50 m | 0.63 | 0.63 | 0.64 |
| 100 m | 1.17 | 1.19 | 1.29 |
| 200 m | 0.79 | 0.83 | 0.84 |
| 500 m | 0.23 | 0.29 | 0.27 |
| 1000 m | 0.15 | 0.17 | 0.16 |

The MLP (with centred targets + tuned hyperparameters) is now on par with the tree ensembles
everywhere and best at the surface (0 m), which is exactly the level surface satellites
inform. The in-app comparison (RMSE on the held-out test split) shows the same story:
**OceanEmbed MLP 0.94 °C vs. linear regression 1.02 °C vs. climatology 1.66 °C** — a
clearly non-trivial gain over the two naive baselines.

Skill is worst at 100–200 m — the main thermocline, where SST alone carries the least
information; below the thermocline (500–1000 m) the profile is near-climatological and
error drops. Adding SSH/SSS (CMEMS + SMAP credentials) is expected to help most at the
thermocline level.

## Demo app

```bash
streamlit run app/streamlit_app.py
```

Click anywhere on the map → the app builds the surface feature vector from **cached**
latest satellite grids (no network at runtime), runs the trained MLP, and plots the
predicted 0–1000 m profile, overlaying the nearest real Argo profile when available.
The sidebar states which sources are live vs skipped/offline. Ship only after
`train.py` has produced `models/deep_mlp.keras`.

### Offline cached snapshots (Cloud demo)

This repo ships the runtime artifacts it needs to render the demo **with zero network
access**, so the app works on Streamlit Community Cloud without storage/API setup:

| Artifact | Contents |
|----------|----------|
| `data/cached/*.nc` | zlib-compressed offline grids for the SST/SSH/SSS replay series (≈ 27 MB total) |
| `data/interim/collocated.parquet` | 3,221 real Argo ↔ satellite collocations (98 floats, 2024-09-01 → 2025-09-30) |
| `data/processed/surface_*_latest.parquet` | latest-frame feature cache for the map |
| `models/deep_mlp.keras`, `scaler_input.joblib`, `feature_meta.json` | trained model + expected feature set |

Manifests (`data/raw/manifest_*.json`) store **repo-relative** paths so the snapshot
ports cleanly between machines (`src/utils/io.py` refuses paths escaping the checkout).
`data/raw/argo_profiles.parquet` and the raw OISST/CMEMS/SMAP `.nc` files are *not*
committed (regenerable); the sidebar badges degrade gracefully and the app still runs.

## Tests

```bash
pytest
```

Fast, network-free unit tests for the collocation KDTree/depth-interpolation logic and
for the model layer (leakage-free float split, feature/target shapes, tiny train/predict
round-trips).

## Known limitations (MVP)

* **Region/window limited** to keep free-tier compute feasible; expanding to the global
  ocean is a config change + wall-time increase, not a code change.
* **`temp_0m` clamp** — Argo's shallowest bin is ~5 m; as in the Argo best-practices
  literature the surface target may use the shallowest real observation.
* **Degraded-mode features:** without CMEMS/Earthdata credentials the pipeline trains on
  `[lat, lon, month, sst]` only. SSH/SSS re-enable automatically once `.env` is filled
  and the sources are re-ingested.
* **Streamlit demo is offline-only for surface fields** (cached latest grids); it does not
  retrain and does not call live APIs.
* **CNN-LSTM patch model off by default** — building block shipped, wiring into the
  collocation pipeline is the next milestone.

## Next milestone after this MVP

1. Wire the CNN-LSTM patch sampler into collocation (spatio-temporal SST patches per
   float) and enable `use_cnn_lstm`.
2. Extend the region/time window to the full Indo-Pacific and add CMEMS SSH + SMAP SSS
   for every deployment (pipeline already supports it).
3. Multi-task/weighted per-depth losses, quantile output for uncertainty, and an
   observational-uncertainty-aware validation study against Argo.
4. A live-mode `streamlit` option that pulls the latest OISST frame in near-real-time.

## Credits & data acknowledgements

* **Argo**: Argo float data and metadata are made available under the Argo Data Policy;
  see http://www.argo.ucsd.edu and http://argo.jcommops.org. Access via the Ifremer GDAC /
  Argovis (University of Colorado).
* **NOAA OISST v2.1**: Reynolds et al. (2007) / Banzon et al. (2020), NOAA NCEI,
  https://www.ncei.noaa.gov/products/optimum-interpolation-sst — retrieved via public ERDDAP.
* **Copernicus Marine** (CMEMS): Sea-level products via https://data.marine.copernicus.eu
  (user licence agreement applies; free of charge).
* **NASA SMAP**: sea-surface salinity L3 products, JPL/PO.DAAC, https://podaac.jpl.nasa.gov
  (NASA Earthdata Login; free of charge).

For this hackathon prototype, **no commercial or paid API** is used — only the open,
free-token systems listed above.