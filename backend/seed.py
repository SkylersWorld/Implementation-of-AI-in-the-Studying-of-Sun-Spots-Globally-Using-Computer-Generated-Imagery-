"""Seed the TSU Observatory (Tbilisi, Georgia) station with the 2026 internship counts.

    python -m backend.seed [--csv data/samples/tsu_tbilisi_2026.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import select

from backend.config import BASE_DIR
from backend.database import SessionLocal, Station, init_db
from backend.stations import import_csv_text

TSU = {"name": "TSU Observatory", "country": "Georgia", "city": "Tbilisi", "latitude": 41.72, "longitude": 44.78, "k_factor": 1.5}


def seed(csv_path: Path) -> tuple[Station, int]:
    init_db()
    with SessionLocal() as db:
        st = db.scalars(select(Station).where(Station.name == TSU["name"])).first()
        if st is None:
            st = Station(**TSU)
            db.add(st)
            db.commit()
            db.refresh(st)
        n = import_csv_text(db, st, csv_path.read_text())
        return st, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=BASE_DIR / "data" / "samples" / "tsu_tbilisi_2026.csv")
    a = p.parse_args()
    st, n = seed(a.csv)
    print(f"Seeded {n} daily counts for {st.name} ({st.city}, {st.country})")


if __name__ == "__main__":
    main()
