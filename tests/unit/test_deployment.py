"""Unit tests for deployment web server and endpoints."""

from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient

from deployment.app import app


@pytest.fixture
def client():
    # Standard Starlette TestClient with anyio/httpx or simple ASGI test
    try:
        from starlette.testclient import TestClient as StarletteClient
        return StarletteClient(app)
    except Exception:
        pytest.skip("TestClient dependencies not fully available for in-memory ASGI test")


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["num_classes"] == 3
    assert "Tumor" in data["classes"]


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "PancreasAI Clinical Suite" in response.text
    assert "Academic & Research Disclaimer" in response.text


def test_static_assets(client):
    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    res_js = client.get("/app.js")
    assert res_js.status_code == 200
