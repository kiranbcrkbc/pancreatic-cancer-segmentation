"""Test public endpoint from outside via HTTPS."""

import io
import json
import urllib.request
import numpy as np
from PIL import Image

PUBLIC_URL = "https://international-diane-wants-goto.trycloudflare.com"

def test_public():
    print(f"Testing public URL: {PUBLIC_URL}")

    # 1. Health check
    h_url = f"{PUBLIC_URL}/health"
    with urllib.request.urlopen(h_url) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"[PASS] /health returned HTTP 200: {data}")
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    # 2. Main page
    with urllib.request.urlopen(PUBLIC_URL) as res:
        assert res.status == 200
        html = res.read().decode()
        print(f"[PASS] / returned HTTP 200 (Length: {len(html)} bytes)")
        assert "PancreasAI Clinical Suite" in html
        assert "Academic & Research Disclaimer" in html

    # 3. Static assets
    with urllib.request.urlopen(f"{PUBLIC_URL}/style.css") as res:
        assert res.status == 200
        print(f"[PASS] /style.css returned HTTP 200")

    with urllib.request.urlopen(f"{PUBLIC_URL}/app.js") as res:
        assert res.status == 200
        print(f"[PASS] /app.js returned HTTP 200")

    # 4. Public Inference
    arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test_slice.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{PUBLIC_URL}/api/predict_slice",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        pred_data = json.loads(res.read().decode())
        print(f"[PASS] /api/predict_slice returned HTTP 200")
        print(f"       has_tumor: {pred_data.get('has_tumor')}")
        print(f"       has_pancreas: {pred_data.get('has_pancreas')}")
        print(f"       tumor_pixels: {pred_data.get('tumor_pixels')}")
        print(f"       pancreas_pixels: {pred_data.get('pancreas_pixels')}")
        print(f"       predicted_mask base64 length: {len(pred_data.get('predicted_mask', ''))}")
        print(f"       gradcam_heatmap base64 length: {len(pred_data.get('gradcam_heatmap', ''))}")

    print("\nALL PUBLIC TESTS OVER HTTPS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_public()
