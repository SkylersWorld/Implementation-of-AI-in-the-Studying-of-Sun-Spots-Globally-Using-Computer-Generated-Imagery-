# Implementation of AI in the Studying of Sun Spots Globally Using Computer-Generated Imagery

**Solar AI Pipeline** — an end-to-end system that ingests full-disk solar imagery (SDO/HMI or ground
telescopes), segments sunspots with a PyTorch U-Net trained on a CGI (computer-generated imagery)
solar disk library, computes the corrected Wolf/International sunspot number

$$R = k\,(10g + s), \qquad k = 1.146$$

(where *g* = groups, *s* = individual spots), attaches seeing-aware uncertainty, falls back to
cubic-spline interpolation when an observation is unreliable, raises Slack/SMS alerts for
δ-class regions and rapid growth, and visualises everything in a live Streamlit dashboard
benchmarked against SILSO.

## Architecture

```mermaid
flowchart LR
    subgraph Sources
        HMI[SDO/HMI via SunPy Fido<br/>continuum + LOS magnetogram]
        GT[Ground telescope frames<br/>PNG / FITS]
        SEE[Scintillation seeing sensor]
    end

    subgraph Ingestion["backend/ingestion"]
        CRON[APScheduler<br/>hourly cron] --> FETCH[hmi_fetcher.py]
        FETCH --> HOT[(Hot folder)]
        GT --> HOT
        HOT --> PROC[processor.py]
    end

    subgraph Model["backend/model"]
        UNET[U-Net segmentation<br/>inference.py] --> COUNT[Count g, s<br/>R = 1.146·(10g+s)]
        CGI[(CGI disk library<br/>image/mask pairs)] --> TRAIN[train.py<br/>BCE · train/val · weights.pt]
        TRAIN -. weights .-> UNET
    end

    subgraph Calibration["backend/calibration"]
        SIG[σ estimate<br/>counting + model + seeing]
        SPL[Cubic-spline fallback<br/>if σ > threshold]
        SILSO[SILSO daily SN<br/>rolling MAE]
    end

    PROC --> UNET
    SEE --> SIG
    COUNT --> SIG --> SPL --> DB[(SQLAlchemy<br/>Observation)]
    SILSO --> DB
    DB --> ALERT[backend/alerts<br/>δ-class · dR/dt → Slack / Twilio]
    DB --> API[FastAPI<br/>backend/main.py]
    API --> DASH[Streamlit dashboard<br/>light curve · CME light · MAE]
    API --> MOUNT[backend/hardware<br/>mount socket sync]
    HMI --> FETCH
```

### Processing steps

1. **Ingestion** – `backend/ingestion/scheduler.py` runs `hourly_job` on an APScheduler cron
   (`INGEST_CRON_MINUTE`, default `:00`). It downloads the latest HMI continuum and LOS magnetogram
   with SunPy `Fido` into the hot folder, then `processor.process_hot_folder` loads every FITS/PNG,
   pairs magnetograms to continuum images (±15 min), processes them, and moves inputs to
   `PROCESSED_FOLDER`.
2. **Segmentation** – `backend/model/inference.py` resizes to `MODEL_INPUT_SIZE`, runs the U-Net,
   thresholds the probability map, labels spots (8-connected, `MIN_SPOT_AREA_PX`), and merges spots
   within `GROUP_MERGE_RADIUS_PX` into groups. If no trained weights exist it falls back to a
   limb-darkening-corrected intensity threshold so the pipeline still runs.
3. **δ-class detection** – with a co-aligned magnetogram, any spot (shared penumbra) containing
   both polarities above 100 G is flagged as a Mount Wilson δ configuration.
4. **Calibration** – `backend/calibration/` combines a Poisson counting term, a model-confidence
   term, and a seeing term (from the scintillation sensor) into σ. If `σ > SIGMA_THRESHOLD`, R is
   replaced by a natural cubic-spline estimate from recent history and the row is flagged
   `interpolated=true`.
5. **Validation** – a rolling MAE of daily-mean R vs. the SILSO daily total sunspot number is
   stored with each observation.
6. **Alerts** – `backend/alerts/` posts to a Slack incoming webhook and/or Twilio SMS when a
   δ-class region appears or `|dR/dt| ≥ DRDT_THRESHOLD` per hour. The CME traffic light is
   green (quiet), amber (δ *or* fast dR/dt), red (both).
7. **Hardware** – `backend/hardware/mount.py` syncs a telescope mount to the Sun (or given RA/Dec)
   using LX200 commands over TCP (`:Sr`, `:Sd`, `:CM#`).

## Repository layout

```
backend/
├── main.py              FastAPI app & routes
├── config.py            pydantic-settings (env / .env)
├── database.py          SQLAlchemy engine + Observation model
├── schemas.py           Pydantic request/response models
├── ingestion/
│   ├── hmi_fetcher.py   SunPy Fido SDO/HMI download → hot folder
│   ├── loader.py        FITS (SunPy/Astropy) / PNG / NPY loading
│   ├── processor.py     image → inference → calibration → DB → alerts
│   └── scheduler.py     APScheduler hourly cron
├── model/
│   ├── unet.py          U-Net
│   ├── dataset.py       CGI library dataset (images/ + masks/)
│   ├── cgi.py           procedural CGI solar disk generator
│   ├── inference.py     segmentation, g/s counting, Wolf number, δ detection
│   └── train.py         training loop → backend/model/weights.pt
├── calibration/
│   ├── seeing.py        scintillation seeing sensor & σ model
│   ├── interpolation.py cubic-spline fallback
│   └── silso.py         SILSO reference & rolling MAE
├── alerts/
│   ├── notifier.py      Slack webhook / Twilio SMS
│   └── rules.py         δ-class & dR/dt rules, CME risk
└── hardware/
    └── mount.py         telescope mount socket sync (LX200 stub)
frontend/dashboard.py    Streamlit dashboard
tests/                   pytest suite
```

## Setup

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # or a CUDA wheel
pip install -r requirements.txt
cp .env.example .env            # edit alert credentials, thresholds, etc.

uvicorn backend.main:app --reload                 # API on :8000 (docs at /docs)
streamlit run frontend/dashboard.py               # dashboard on :8501
python -m backend.ingestion.scheduler             # hourly HMI ingest (or SCHEDULER_ENABLED=true)
```

### Docker

```bash
cp .env.example .env
docker compose up --build
```

Services: `db` (Postgres 16), `api` (:8000), `scheduler` (hourly ingest), `dashboard` (:8501).
`./data` and `./backend/model` are mounted so the hot folder and weights persist.

### Training the U-Net

Place the CGI solar disk library as matching image/mask pairs:

```
data/cgi/images/disk_0001.png    data/cgi/masks/disk_0001.png
```

Masks are binary (non-zero = sunspot pixel). PNG/JPEG/TIFF/NPY are supported.

```bash
# Optional: synthesise a CGI library with the bundled procedural renderer
python -m backend.model.cgi --out data/cgi -n 1000 --size 256

python -m backend.model.train --data data/cgi --epochs 30 --batch-size 8 --size 256 \
    --val-split 0.2 --pos-weight 3 --out backend/model/weights.pt
```

The loop uses `BCEWithLogitsLoss`, a seeded train/val split, Adam + ReduceLROnPlateau, logs
loss/Dice/IoU per epoch, saves the best-validation checkpoint to `backend/model/weights.pt`
(plus `weights_last.pt` and a `.history.json`). The API picks the weights up on start.

### Tests

```bash
pytest
```

## API

Interactive docs: `http://localhost:8000/docs`.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Status and whether the U-Net or classical fallback is active |
| GET | `/observations?limit=500&since=ISO8601` | Observations, oldest → newest |
| GET | `/observations/latest` | Most recent observation (404 if none) |
| GET | `/observations/{id}` | Single observation |
| POST | `/observations` | Insert an observation manually (JSON body, `ObservationCreate`) |
| POST | `/infer` | Multipart upload: `image` (FITS/PNG/JPEG/TIFF/NPY), optional `magnetogram`, `seeing_arcsec`, `timestamp`, `source`. Runs the full pipeline and stores the result |
| POST | `/ingest/process?seeing_arcsec=` | Process everything currently in the hot folder |
| POST | `/ingest/run` | Trigger the hourly HMI fetch + process job now (async, 202) |
| POST | `/calibration/seeing` | `{"samples": [..]}` photometer intensities → scintillation index, seeing (″), Fried r₀ (cm) |
| GET | `/metrics` | `count`, `latest_r`, `rolling_mae`, `drdt_per_hour`, `cme_risk` (`green`/`amber`/`red`) |
| GET | `/metrics/silso?days=90` | SILSO daily reference series |
| POST | `/hardware/sync` | `{"ra_deg", "dec_deg"}` or empty body to sync the mount to the Sun; 502 if unreachable |

### Observation schema

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | int | Primary key |
| `timestamp` | datetime (UTC) | Observation time |
| `source` | str | `hmi`, `ground`, `upload`, … |
| `filename` | str? | Input file |
| `groups` | int | Sunspot groups *g* |
| `spots` | int | Individual spots *s* |
| `wolf_raw` | float | 10g + s |
| `r_corrected` | float | k·(10g + s), or spline estimate if interpolated |
| `sigma` | float | 1σ uncertainty of R |
| `confidence` | float | Mean model probability over detected spots (0–1) |
| `delta_class` | bool | δ-class region present |
| `mae` | float? | Rolling MAE vs SILSO at insert time |
| `seeing_arcsec` | float? | Seeing from the scintillation sensor |
| `interpolated` | bool | R replaced by cubic-spline fallback |

### Example

```bash
curl -F image=@hmi_ic_45s.fits -F magnetogram=@hmi_m_45s.fits -F seeing_arcsec=1.2 \
     http://localhost:8000/infer
```

## Configuration

All settings live in `backend/config.py` and can be overridden via environment variables or
`.env` (see `.env.example`), e.g. `SIGMA_THRESHOLD`, `WOLF_K_FACTOR`, `DRDT_THRESHOLD`,
`SLACK_WEBHOOK_URL`, `TWILIO_*`, `MOUNT_HOST`/`MOUNT_PORT`.

## License

MIT — see [LICENSE](LICENSE).
