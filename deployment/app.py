"""
Clinical Demonstration Web Application Backend
FastAPI server serving clinical segmentation inference, Grad-CAM overlays, and health status.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.models.cbam_net import CBAMNet
from src.models.cnn_mhsa import CNNMHSASeg
from src.models.att_unet_gat import AttnUNetEfficientGAT
from src.xai.gradcam import GradCAMSeg
from src.preprocessing.ct_preprocessor import CTPreprocessor
from src.preprocessing.roi_extractor import ROIPatchExtractor


app = FastAPI(
    title="Pancreatic Cancer Segmentation Clinical Dashboard",
    description="Automated 3-class segmentation & XAI diagnostic viewer supporting 4 models",
    version="2.0.0",
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

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pyramid": {
        "class": CNNPyramidTransformerSeg,
        "name": "Model 1: CNN + Pyramid Transformer",
        "ckpt": "checkpoints/pyramid/final_model.pt",
        "fallback_ckpt": "checkpoints/final_model.pt",
    },
    "cbam": {
        "class": CBAMNet,
        "name": "Model 2: CNN + CBAM",
        "ckpt": "checkpoints/cbam/final_model.pt",
        "fallback_ckpt": "checkpoints/cbam/best_model_fold_1.pt",
    },
    "mhsa": {
        "class": CNNMHSASeg,
        "name": "Model 3: CNN + MHSA",
        "ckpt": "checkpoints/mhsa/final_model.pt",
        "fallback_ckpt": "checkpoints/mhsa/best_model_fold_1.pt",
    },
    "gnn": {
        "class": AttnUNetEfficientGAT,
        "name": "Model 4: CNN + GNN/GAT (EfficientNet-B3 + 4L GAT)",
        "ckpt": "checkpoints/gnn/final_model.pt",
        "fallback_ckpt": "checkpoints/gnn/best_model_fold_1.pt",
    },
}

LOADED_MODELS: Dict[str, nn.Module] = {}

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

def get_loaded_model(model_key: str = "pyramid") -> tuple[nn.Module, str]:
    key = (model_key or "pyramid").lower().strip()
    if key not in MODEL_REGISTRY:
        key = "pyramid"

    if key in LOADED_MODELS:
        return LOADED_MODELS[key], key

    info = MODEL_REGISTRY[key]
    model_instance = info["class"](in_channels=1, num_classes=3).to(device).eval()
    for param in model_instance.parameters():
        param.requires_grad = False

    ckpt_path = Path(info["ckpt"])
    if not (ckpt_path.exists() and ckpt_path.stat().st_size > 1000):
        ckpt_path = Path(info["fallback_ckpt"])
    if not (ckpt_path.exists() and ckpt_path.stat().st_size > 1000) and key == "pyramid":
        ckpt_path = ensure_model_checkpoint()

    if ckpt_path.exists() and ckpt_path.stat().st_size > 1000:
        try:
            data = torch.load(ckpt_path, map_location=device)
            state_dict = data.get("model_state_dict", data)
            model_instance.load_state_dict(state_dict)
            print(f"Successfully loaded {key} model weights from {ckpt_path}.", flush=True)
        except Exception as e:
            print(f"Warning: Failed to load {key} checkpoint ({e}). Using initialized weights.", flush=True)

    LOADED_MODELS[key] = model_instance
    return model_instance, key

# Pre-load default Pyramid model
default_model, _ = get_loaded_model("pyramid")

preprocessor = CTPreprocessor()
roi_extractor = ROIPatchExtractor(patch_size=(128, 128))


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": len(LOADED_MODELS) > 0,
        "num_classes": 3,
        "classes": ["Background", "Pancreas", "Tumor"],
        "available_models": list(MODEL_REGISTRY.keys()),
        "default_model": "pyramid",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    template_path = Path("deployment/templates/index.html")
    if not template_path.exists():
        template_path = Path("deployment/index.html")
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "<h1>Pancreatic Cancer Segmentation API Active</h1><p>Visit /docs for API documentation.</p>"


@app.get("/model/{model_key}", response_class=HTMLResponse)
def model_page(model_key: str):
    """Direct landing endpoint for a specific model."""
    template_path = Path("deployment/templates/index.html")
    if not template_path.exists():
        template_path = Path("deployment/index.html")
    if template_path.exists():
        html = template_path.read_text(encoding="utf-8")
        # Ensure the selector defaults to requested model if valid
        key = model_key.lower().strip()
        if key in MODEL_REGISTRY:
            html = html.replace('selected>Model 1: CNN + Pyramid Transformer (PPM + MHSA)</option>', '>Model 1: CNN + Pyramid Transformer (PPM + MHSA)</option>')
            html = html.replace(f'value="{key}"', f'value="{key}" selected')
        return html
    return f"<h1>Model {model_key} Active</h1>"


@app.get("/api/models")
def get_models_info():
    """Returns registry and operational details for all four models."""
    res = {}
    for k, v in MODEL_REGISTRY.items():
        res[k] = {
            "key": k,
            "name": v["name"],
            "checkpoint": v["ckpt"],
            "checkpoint_exists": Path(v["ckpt"]).exists(),
            "loaded": k in LOADED_MODELS,
        }
    return res


@app.get("/api/comparison")
def get_comparison_metrics():
    """Returns the four-model comparison JSON if available."""
    comp_file = Path("results/four_model_comparison.json")
    if comp_file.exists():
        import json
        with open(comp_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "Compilation pending or running"}


@app.post("/api/predict_slice")
async def predict_slice(
    file: UploadFile = File(...),
    model_type: str = Form("pyramid"),
):
    """Accepts image slice upload and model selection, runs segmentation and Grad-CAM, returns base64 images."""
    content = await file.read()
    if not content or len(content) == 0:
        return JSONResponse(status_code=400, content={"error": "Uploaded file is empty"})
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("L")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image format: {str(e)}"})

    active_model, used_key = get_loaded_model(model_type)

    img_np = np.array(pil_img, dtype=np.float32) / 255.0

    proc_sl = preprocessor.process_slice(img_np)
    patch_img, _, _ = roi_extractor.extract_patch(proc_sl)

    t = torch.from_numpy(patch_img).unsqueeze(0).unsqueeze(0).float().to(device)

    # 1. Forward Pass
    with torch.no_grad():
        logits = active_model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        preds = np.argmax(probs, axis=0)

    # 2. Grad-CAM for Tumor (Class 2)
    cam_layer = active_model.get_cam_target_layer()
    cam_engine = GradCAMSeg(active_model, cam_layer)
    cam_heatmap = cam_engine.generate_heatmap(t, target_class=2)
    cam_engine.close()

    # Convert to base64 images
    def to_b64(arr: np.ndarray, cmap_name: str = "gray", vmin: float = 0.0, vmax: float = 1.0) -> str:
        diff = max(vmax - vmin, 1e-6)
        if cmap_name == "gray":
            norm = np.clip((arr - vmin) / diff * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(norm, mode="L")
        else:
            norm_arr = np.clip((arr - vmin) / diff, 0.0, 1.0)
            cmap = plt.get_cmap(cmap_name)
            rgba = (cmap(norm_arr) * 255).astype(np.uint8)
            pil = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    img_b64 = to_b64(patch_img, cmap_name="gray", vmin=0.0, vmax=1.0)
    pred_b64 = to_b64(preds.astype(float), cmap_name="viridis", vmin=0.0, vmax=2.0)
    cam_b64 = to_b64(cam_heatmap, cmap_name="jet", vmin=0.0, vmax=1.0)

    # Metrics on patch
    has_tumor = bool(np.any(preds == 2))
    has_panc = bool(np.any(preds == 1))
    panc_pixels = int(np.sum(preds == 1))
    tumor_pixels = int(np.sum(preds == 2))

    # Free temporary tensors and garbage collect
    del t, logits, probs, cam_engine, cam_heatmap
    import gc
    gc.collect()

    return {
        "model_used": used_key,
        "model_name": MODEL_REGISTRY[used_key]["name"],
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
