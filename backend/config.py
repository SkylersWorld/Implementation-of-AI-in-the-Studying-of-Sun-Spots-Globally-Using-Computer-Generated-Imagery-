"""Application settings loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Core
    app_name: str = "Solar AI Pipeline"
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'solar.db'}"
    api_base_url: str = "http://localhost:8000"

    # Ingestion
    hot_folder: Path = BASE_DIR / "data" / "hot_folder"
    processed_folder: Path = BASE_DIR / "data" / "processed"
    hmi_series: str = "hmi.Ic_45s"
    hmi_lookback_minutes: int = 60
    ingest_cron_minute: str = "0"
    scheduler_enabled: bool = False

    # Model
    model_weights: Path = BASE_DIR / "backend" / "model" / "weights.pt"
    model_input_size: int = 512
    seg_threshold: float = 0.5
    min_spot_area_px: int = 4
    group_merge_radius_px: int = 12
    wolf_k_factor: float = 1.146

    # Calibration
    sigma_threshold: float = 15.0
    interpolation_window: int = 48
    seeing_coeff_a: float = 0.5
    seeing_coeff_b: float = 0.8
    seeing_sigma_per_arcsec: float = 0.04
    silso_enabled: bool = True
    silso_url: str = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
    swpc_url: str = "https://services.swpc.noaa.gov/json/solar-cycle/swpc_observed_ssn.json"
    helioviewer_url: str = "https://api.helioviewer.org/v2/takeScreenshot/"
    daily_auto_analyze: bool = True
    daily_image_folder: Path = BASE_DIR / "data" / "daily"
    daily_analysis_size: int = 2048
    daily_contrast: float = 0.9
    daily_umbra_contrast: float = 0.65
    daily_group_radius_px: int = 6
    mae_window: int = 30

    # Alerts
    slack_webhook_url: str | None = None
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from_number: str | None = None
    twilio_to_number: str | None = None
    drdt_threshold: float = 20.0  # R units per hour

    # Hardware
    mount_host: str = "127.0.0.1"
    mount_port: int = 9999
    mount_timeout_s: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
