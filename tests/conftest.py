import pytest

from medrag.config import ROOT_DIR, Settings
from medrag.retrieval import RetrievalPipeline

# Unit tests run against the small, stable sample corpus; the PubMed Central corpus has its own tests.
SAMPLE_KB = [ROOT_DIR / "data" / "knowledge_base"]


@pytest.fixture
def settings(tmp_path):
    return Settings(mode="offline", data_dir=tmp_path / "var", seed_demo=True, secret_key="test-secret",
                    patient_salt="test-salt", kb_dirs=list(SAMPLE_KB))


@pytest.fixture(scope="session")
def pipeline():
    p = RetrievalPipeline(Settings(mode="offline", kb_dirs=list(SAMPLE_KB)))
    p.ingest_directory()
    return p


@pytest.fixture
def service(settings):
    from medrag.service import MedRAGService

    return MedRAGService(settings)


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    from medrag.api import create_app

    return TestClient(create_app(settings))


def login(client, user="clinician", pw=None):
    pw = pw or {"clinician": "clinician123", "admin": "admin123", "viewer": "viewer123"}[user]
    r = client.post("/api/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}
