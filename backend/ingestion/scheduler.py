"""APScheduler hourly cron: fetch HMI data into the hot folder and process it.

Run standalone with ``python -m backend.ingestion.scheduler``.
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.config import get_settings
from backend.database import init_db
from backend.ingestion.hmi_fetcher import fetch_hourly_bundle
from backend.ingestion.processor import process_hot_folder

log = logging.getLogger(__name__)
JOB_ID = "hourly_hmi_ingest"


def hourly_job() -> int:
    fetch_hourly_bundle()
    observations = process_hot_folder()
    log.info("Hourly ingest processed %d observation(s)", len(observations))
    return len(observations)


def _add_job(scheduler) -> None:
    scheduler.add_job(
        hourly_job,
        CronTrigger(minute=get_settings().ingest_cron_minute),
        id=JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )


def start_background() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    _add_job(scheduler)
    scheduler.start()
    log.info("Background scheduler started (cron minute=%s)", get_settings().ingest_cron_minute)
    return scheduler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    init_db()
    scheduler = BlockingScheduler(timezone="UTC")
    _add_job(scheduler)
    log.info("Running initial ingest, then hourly at minute %s", get_settings().ingest_cron_minute)
    try:
        hourly_job()
    except Exception:
        log.exception("Initial ingest failed")
    scheduler.start()


if __name__ == "__main__":
    main()
