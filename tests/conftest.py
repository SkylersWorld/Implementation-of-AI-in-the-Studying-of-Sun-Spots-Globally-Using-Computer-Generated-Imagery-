import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="solar-ai-tests-"))
os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
        "HOT_FOLDER": str(_TMP / "hot"),
        "PROCESSED_FOLDER": str(_TMP / "processed"),
        "MODEL_WEIGHTS": str(_TMP / "no-weights.pt"),
        "SILSO_ENABLED": "false",
        "SCHEDULER_ENABLED": "false",
        "MODEL_INPUT_SIZE": "256",
        "SLACK_WEBHOOK_URL": "",
        "TWILIO_ACCOUNT_SID": "",
    }
)

import pytest  # noqa: E402

from backend.database import Base, engine, init_db  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def tmp_root():
    return _TMP
