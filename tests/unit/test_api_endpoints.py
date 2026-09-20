"""Comprehensive API & deployment endpoint tests covering items O-W."""

import io
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
from starlette.testclient import TestClient

from deployment.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_o_deployment_application(client):
    # O. Deployment application initialized
    assert app is not None
    assert app.title == "Pancreatic Cancer Segmentation Clinical Dashboard"


def test_p_api_endpoints_status(client):
    # P. API endpoints respond
    res_root = client.get("/")
    assert res_root.status_code == 200
    res_health = client.get("/health")
    assert res_health.status_code == 200


def test_q_frontend_assets(client):
    # Q. Frontend assets load
    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    assert "font-family" in res_css.text or "color" in res_css.text

    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert "addEventListener" in res_js.text


def test_r_s_file_upload_and_prediction(client):
    # R & S. File upload & prediction endpoint with valid PNG
    arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test_slice.png", buf, "image/png")}
    response = client.post("/api/predict_slice", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "has_tumor" in data
    assert "has_pancreas" in data
    assert "predicted_mask" in data
    assert data["predicted_mask"].startswith("data:image/png;base64,")
    assert "gradcam_heatmap" in data


def test_t_invalid_input_handling(client):
    # T. Invalid input handling
    # 1. Empty file
    empty_files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
    res_empty = client.post("/api/predict_slice", files=empty_files)
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["error"].lower()

    # 2. Corrupt bytes
    corrupt_files = {"file": ("corrupt.txt", io.BytesIO(b"not an image data string"), "text/plain")}
    res_corrupt = client.post("/api/predict_slice", files=corrupt_files)
    assert res_corrupt.status_code == 400
    assert "invalid image" in res_corrupt.json()["error"].lower()


def test_w_health_endpoint_schema(client):
    # W. Health endpoint
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 3
    assert data["classes"] == ["Background", "Pancreas", "Tumor"]
