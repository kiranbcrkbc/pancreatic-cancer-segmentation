"""
Clinical Demonstration Web Application Backend
FastAPI server serving clinical segmentation inference, Grad-CAM overlays, and health status.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.xai.gradcam import GradCAMSeg
from src.preprocessing.ct_preprocessor import CTPreprocessor
from src.preprocessing.roi_extractor import ROIPatchExtractor


app = FastAPI(
    title="Pancreatic Cancer Segmentation Clinical Dashboard",
    description="Automated 3-class segmentation & XAI diagnostic viewer",
    version="1.0.0",
)

# Static files
static_dir = Path("deployment/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/style.css")
def get_style():
    p = Path("deployment/style.css")
    if not p.exists():
        p = Path("deployment/static/css/style.css")
    return HTMLResponse(content=p.read_text(encoding="utf-8"), media_type="text/css")

@app.get("/app.js")
def get_script():
    p = Path("deployment/app.js")
    if not p.exists():
        p = Path("deployment/static/js/app.js")
    return HTMLResponse(content=p.read_text(encoding="utf-8"), media_type="application/javascript")

# Device and Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
model = CNNPyramidTransformerSeg(in_channels=1, num_classes=3)

def ensure_model_checkpoint() -> Path:
    target_path = Path("checkpoints/final_model.pt")
    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path
    fold1 = Path("checkpoints/fold1_best.pt")
    if fold1.exists() and fold1.stat().st_size > 1000:
        return fold1
    release_url = "https://github.com/kiranbcrkbc/pancreatic-cancer-segmentation/releases/download/v1.0.0/final_model.pt"
    print(f"Fetching production checkpoint from {release_url}...", flush=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        urllib.request.urlretrieve(release_url, str(target_path))
        print("Model checkpoint fetched successfully.", flush=True)
        return target_path
    except Exception as e:
        print(f"Warning: could not download model weights: {e}", flush=True)
        return target_path

ckpt_path = ensure_model_checkpoint()
if ckpt_path.exists() and ckpt_path.stat().st_size > 1000:
    try:
        data = torch.load(ckpt_path, map_location=device)
        state_dict = data.get("model_state_dict", data)
        model.load_state_dict(state_dict)
        del data, state_dict
        import gc
        gc.collect()
        print("Production model weights successfully loaded into CNNPyramidTransformerSeg.", flush=True)
    except Exception as e:
        print(f"Warning: Model state load error: {e}", flush=True)

model.to(device).eval()
preprocessor = CTPreprocessor()
roi_extractor = ROIPatchExtractor(patch_size=(128, 128))


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": ckpt_path.exists() and (ckpt_path.stat().st_size > 1000),
        "num_classes": 3,
        "classes": ["Background", "Pancreas", "Tumor"],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    template_path = Path("deployment/templates/index.html")
    if not template_path.exists():
        template_path = Path("deployment/index.html")
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "<h1>Pancreatic Cancer Segmentation API Active</h1><p>Visit /docs for API documentation.</p>"


@app.post("/api/predict_slice")
async def predict_slice(file: UploadFile = File(...)):
    """Accepts image slice upload, runs segmentation and Grad-CAM, returns base64 images."""
    content = await file.read()
    if not content or len(content) == 0:
        return JSONResponse(status_code=400, content={"error": "Uploaded file is empty"})
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("L")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image format: {str(e)}"})

    img_np = np.array(pil_img, dtype=np.float32) / 255.0

    proc_sl = preprocessor.process_slice(img_np)
    patch_img, _, _ = roi_extractor.extract_patch(proc_sl)

    t = torch.from_numpy(patch_img).unsqueeze(0).unsqueeze(0).float().to(device)

    # 1. Forward Pass
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        preds = np.argmax(probs, axis=0)

    # 2. Grad-CAM for Tumor (Class 2)
    cam_engine = GradCAMSeg(model, model.get_cam_target_layer())
    cam_heatmap = cam_engine.generate_heatmap(t, target_class=2)
    cam_engine.close()

    # Convert to base64 images
    def to_b64(arr, cmap="gray", vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    img_b64 = to_b64(patch_img, cmap="gray")
    pred_b64 = to_b64(preds, cmap="viridis", vmin=0, vmax=2)
    cam_b64 = to_b64(cam_heatmap, cmap="jet")

    # Metrics on patch
    has_tumor = bool(np.any(preds == 2))
    has_panc = bool(np.any(preds == 1))
    panc_pixels = int(np.sum(preds == 1))
    tumor_pixels = int(np.sum(preds == 2))

    return {
        "has_tumor": has_tumor,
        "has_pancreas": has_panc,
        "pancreas_pixels": panc_pixels,
        "tumor_pixels": tumor_pixels,
        "input_slice": f"data:image/png;base64,{img_b64}",
        "predicted_mask": f"data:image/png;base64,{pred_b64}",
        "gradcam_heatmap": f"data:image/png;base64,{cam_b64}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
