# Implementation of AI in the Studying of Sun Spots Globally Using Computer-Generated Imagery

**Solar AI Pipeline** — an end-to-end system that ingests full-disk solar imagery (SDO/HMI or ground
telescopes), segments sunspots with a PyTorch U-Net trained on a CGI (computer-generated imagery)
solar disk library, computes the corrected Wolf/International sunspot number

$$R = k\,(10g + s), \qquad k = 1.146$$

(where *g* = groups, *s* = individual spots), attaches seeing-aware uncertainty, falls back to
cubic-spline interpolation when an observation is unreliable, raises Slack/SMS alerts for
δ-class regions and rapid growth, and visualises everything in a live Streamlit dashboard
benchmarked against the international sunspot number (SILSO, with NOAA SWPC as fallback).

It also compares **your own observatory's daily counts** (by country) with the international R.
For every day you submit, the AI downloads the real SDO/HMI image of the Sun, counts g and s on it
(annotated image in the dashboard), and your counts are calibrated against the AI's. The dashboard
shows raw, local, calibrated, AI and international R, % differences, MAE, and linear/cubic
interpolation of missing days.

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
        SILSO[International R<br/>manual › SILSO › NOAA SWPC<br/>cached · rolling MAE]
        LOCAL[Station calibration<br/>k_fit · % diff · interpolation]
    end

    PROC --> UNET
    SEE --> SIG
    COUNT --> SIG --> SPL --> DB[(SQLAlchemy<br/>Observation)]
    SILSO --> DB
    OBS[Station daily counts<br/>form / CSV] --> LOCAL --> DB
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
5. **Validation** – a rolling MAE of daily-mean R vs. the international daily sunspot number is
   stored with each observation (see *International R* below).
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
├── database.py          SQLAlchemy engine; Observation, Station, DailyCount, ReferenceSN
├── stations.py          station / daily count / comparison / reference routes
├── schemas.py           Pydantic request/response models
├── ingestion/
│   ├── hmi_fetcher.py   SunPy Fido SDO/HMI download → hot folder
│   ├── loader.py        FITS (SunPy/Astropy) / PNG / NPY loading
│   ├── daily.py         per-day HMI fetch (Fido → Helioviewer) + AI g/s count + annotated PNG
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
│   ├── silso.py         SILSO CSV parser & rolling MAE
│   ├── reference.py     manual › SILSO › NOAA SWPC, cached in reference_sn
│   └── local.py         station calibration: W, k_fit, % diff, interpolation
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
| GET | `/metrics/silso?days=90` | International daily R (best available source) with `source` per day |
| POST | `/hardware/sync` | `{"ra_deg", "dec_deg"}` or empty body to sync the mount to the Sun; 502 if unreachable |

### Stations & international comparison

| Method | Path | Description |
| --- | --- | --- |
| GET | `/countries` | Countries that have a station |
| GET / POST | `/stations[?country=]` | List / create stations (`name`, `country`, `city`, `latitude`, `longitude`, `k_factor`) |
| GET / PATCH / DELETE | `/stations/{id}` | Single station |
| GET / POST | `/stations/{id}/counts` | List / upsert a daily count `{date, groups, spots, k?, seeing_arcsec?, notes?, international_r?}` |
| DELETE | `/stations/{id}/counts/{date}` | Remove a day |
| POST | `/stations/{id}/counts/import` | Multipart CSV `date,g,s[,k,ri,seeing,notes]` (dates `YYYY-MM-DD` or `DD.MM.YY`) |
| GET | `/stations/{id}/compare?start=&end=&source=auto&interpolation=linear` | Daily rows + summary (see below) |
| GET | `/reference?start=&end=&source=auto` | International R with source per day |
| PUT / DELETE | `/reference/{date}` | Set / remove a manual international value `{value}` |
| POST | `/reference/refresh` | Re-download SILSO and NOAA now |
| GET | `/solar-days?start=&end=` | AI counts of the real HMI image per date (`status`, `groups`, `spots`, `r_ai`, `source`) |
| GET | `/solar-days/{date}` / `/solar-days/{date}/image` | One day's AI count / annotated PNG |
| POST | `/solar-days/{date}/analyze?longitude=` | (Re)analyse a day in the background (202) |

For each day the comparison returns

- `wolf_raw` W = 10g + s
- `r_local` = k·W with the observer's k (or the station's default `k_factor`)
- `r_ai`, `ai_groups`, `ai_spots`: the AI's count on the real SDO/HMI image of that day
- `r_calibrated` = k_fit·W, where k_fit = Σ(W·R_AI) / Σ(W²) is the least-squares fit over every
  observed day with an AI count (international R is used only if no day has been analysed)
- `r_international` and `reference_source` (`manual`, `silso` or `swpc`)
- `pct_diff_local`, `pct_diff_calibrated`, `pct_diff_ai` = (R − R_intl) / R_intl × 100

Days between two observations are filled with `R_d = R_a + (d − a)/(b − a)·(R_b − R_a)`
(`interpolation=linear`) or a natural cubic spline, and are returned with `observed=false`.
The summary has k_fit ± σ, the mean and population σ of local R, MAE/RMSE and the mean % difference
(local and calibrated), computed over observed days only.

**International R.** Each date resolves as *manual value › SILSO › NOAA SWPC*. SILSO
(`SN_d_tot_V2.0.csv`) and NOAA (`swpc_observed_ssn.json`) are downloaded at most every 6 h and stored
in the `reference_sn` table, so comparisons and MAE keep working when either site is unreachable.
Values you enter yourself (form, CSV `ri` column, or `PUT /reference/{date}`) always win.

**AI count of the real Sun.** Submitting a count (form, CSV or `POST /stations/{id}/counts`) queues
`backend/ingestion/daily.py` for that date: it downloads the HMI continuum image nearest the
station's local solar noon (12:00 UTC − longitude/15) as a FITS file via SunPy Fido, falling back to the
Helioviewer API, then counts sunspots on a 2048² limb-darkening-corrected image. Spots are umbrae
(intensity < 0.65 of the local disk background) inside penumbral regions (< 0.9); a pore with no umbra
counts once. Spots within a merge radius form a group. If `backend/model/weights.pt` exists, the U-Net
mask is used instead. On 8 days in Aug–Sep 2026 the classical counter's W = 10g + s was within about
15 R (MAE) of NOAA's daily number, and lower on average, because it misses small pores. Each date is
analysed once and shared by all stations. Set `DAILY_AUTO_ANALYZE=false` to switch this off.

**dR/dt** is how fast R changes, in R per hour: (R_now − R_prev) / Δt. Above `DRDT_THRESHOLD`
(default 20/h) it triggers an alert and the amber/red traffic light.

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
