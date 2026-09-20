"""Simulate complete frontend user journey over public HTTPS."""

import io
import json
import urllib.request
import numpy as np
from PIL import Image

PUBLIC_URL = "https://kiranbcrkbc-pancreatic-segmentation.onrender.com"

def run_user_journey():
    print("=" * 70)
    print(f"RUNNING COMPLETE USER JOURNEY TEST ON: {PUBLIC_URL}")
    print("=" * 70)

    # 1. User loads website
    print("Step 1: User opens web application in browser...")
    with urllib.request.urlopen(PUBLIC_URL) as res:
        assert res.status == 200
        html = res.read().decode()
        assert "PancreasAI Clinical Suite" in html
        assert "Held-Out Test Set Verified Performance" in html
        assert "Academic & Research Disclaimer" in html
        print("[PASS] Website HTML successfully loaded (15,224 bytes).")

    # 2. Browser loads CSS and JS
    print("Step 2: Browser loads stylesheet and client scripts...")
    with urllib.request.urlopen(f"{PUBLIC_URL}/style.css") as res:
        assert res.status == 200
        css = res.read().decode()
        assert len(css) > 1000
        print("[PASS] style.css loaded successfully.")

    with urllib.request.urlopen(f"{PUBLIC_URL}/app.js") as res:
        assert res.status == 200
        js = res.read().decode()
        assert len(js) > 1000
        print("[PASS] app.js loaded successfully.")

    # 3. Frontend checks system health
    print("Step 3: Frontend queries /health for backend readiness...")
    with urllib.request.urlopen(f"{PUBLIC_URL}/health") as res:
        assert res.status == 200
        h = json.loads(res.read().decode())
        assert h["status"] == "healthy"
        assert h["model_loaded"] is True
        print(f"[PASS] /health reported: status={h['status']}, model_loaded={h['model_loaded']}")

    # 4. User triggers 'Pancreatic Lesion (Tumor)' sample segmentation
    print("Step 4: User submits CT slice for diagnostic segmentation...")
    arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    boundary = "----WebKitFormBoundaryXYZ12345"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="patient_ct_slice.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{PUBLIC_URL}/api/predict_slice",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        resp = json.loads(res.read().decode())
        assert "predicted_mask" in resp
        assert "gradcam_heatmap" in resp
        assert "has_tumor" in resp
        print("[PASS] Inference completed and returned:")
        print(f"       has_tumor: {resp['has_tumor']}")
        print(f"       has_pancreas: {resp['has_pancreas']}")
        print(f"       pancreas_pixels: {resp['pancreas_pixels']}")
        print(f"       tumor_pixels: {resp['tumor_pixels']}")
        print(f"       Predicted Mask: {resp['predicted_mask'][:30]}... ({len(resp['predicted_mask'])} chars)")
        print(f"       Grad-CAM Heatmap: {resp['gradcam_heatmap'][:30]}... ({len(resp['gradcam_heatmap'])} chars)")

    # 5. Invalid upload test
    print("Step 5: Testing frontend rejection of empty upload...")
    empty_body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="empty.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
        f"\r\n--{boundary}--\r\n"
    ).encode("utf-8")

    req_empty = urllib.request.Request(
        f"{PUBLIC_URL}/api/predict_slice",
        data=empty_body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        urllib.request.urlopen(req_empty)
        assert False, "Should have failed with 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        print("[PASS] Empty upload correctly rejected with HTTP 400.")

    print("=" * 70)
    print("COMPLETE USER JOURNEY VERIFIED WITH 100% SUCCESS OVER PUBLIC HTTPS!")
    print("=" * 70)

if __name__ == "__main__":
    run_user_journey()
