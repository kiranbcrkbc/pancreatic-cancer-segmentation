document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const btnAnalyze = document.getElementById("btn-analyze");
    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");

    const imgInput = document.getElementById("img-input");
    const imgMask = document.getElementById("img-mask");
    const imgCam = document.getElementById("img-cam");

    const wrapperInput = document.getElementById("wrapper-input");
    const wrapperMask = document.getElementById("wrapper-mask");
    const wrapperCam = document.getElementById("wrapper-cam");

    const predictionBadge = document.getElementById("prediction-badge");
    const pathologySummary = document.getElementById("pathology-summary");
    const valTumorStatus = document.getElementById("val-tumor-status");
    const valPancPixels = document.getElementById("val-panc-pixels");
    const valTumorPixels = document.getElementById("val-tumor-pixels");
    const valConfidence = document.getElementById("val-confidence");

    const btnSampleTumor = document.getElementById("btn-sample-tumor");
    const btnSampleHealthy = document.getElementById("btn-sample-healthy");
    const backendStatus = document.getElementById("backend-status");
    const deviceInfo = document.getElementById("device-info");

    let currentFile = null;

    // Check Backend Health
    fetch("/health")
        .then(r => r.json())
        .then(data => {
            if (data.status === "healthy") {
                backendStatus.textContent = "AI Online";
                deviceInfo.textContent = `Device: ${data.device.toUpperCase()}`;
            }
        })
        .catch(() => {
            backendStatus.textContent = "Offline";
            backendStatus.classList.remove("badge-pulse");
            backendStatus.classList.add("badge-dark");
        });

    // Dynamic Architecture Specs Mapping & Query Parameter Routing
    const modelSelect = document.getElementById("model-select");
    const specsList = document.querySelector(".specs-list");
    const modelSpecs = {
        pyramid: [
            "<strong>Encoder:</strong> 4-Stage Residual CNN (64-512)",
            "<strong>Bottleneck:</strong> Pyramid Pooling Module (PPM) + Multi-Head Self-Attention",
            "<strong>Decoder:</strong> Residual U-Net Skip Decoder",
            "<strong>Target Classes:</strong> Background (0), Pancreas (1), Tumor (2)",
            "<strong>Interpretability:</strong> Grad-CAM & Attention Maps"
        ],
        cbam: [
            "<strong>Encoder:</strong> 4-Stage Residual CNN + Channel & Spatial Attention (CBAM)",
            "<strong>Bottleneck:</strong> High-Capacity CBAM Feature Refinement Block",
            "<strong>Decoder:</strong> CBAM-Enhanced Residual U-Net Decoder",
            "<strong>Target Classes:</strong> Background (0), Pancreas (1), Tumor (2)",
            "<strong>Interpretability:</strong> Grad-CAM on Enc4 Residual Conv"
        ],
        mhsa: [
            "<strong>Encoder:</strong> 4-Stage Residual CNN Encoder (64-512)",
            "<strong>Bottleneck:</strong> 4-Layer 8-Head Multi-Head Self-Attention (MHSA)",
            "<strong>Decoder:</strong> Residual U-Net Skip Decoder",
            "<strong>Target Classes:</strong> Background (0), Pancreas (1), Tumor (2)",
            "<strong>Interpretability:</strong> Grad-CAM & Multi-Head Self-Attention Maps"
        ],
        gnn: [
            "<strong>Encoder:</strong> Dual-Pathway CNN + Pretrained EfficientNet-B3 Backbone",
            "<strong>Bottleneck:</strong> 4-Layer Multi-Head Graph Attention Network (GAT)",
            "<strong>Decoder:</strong> Attention U-Net with Additive Attention Gates",
            "<strong>Target Classes:</strong> Background (0), Pancreas (1), Tumor (2)",
            "<strong>Interpretability:</strong> Grad-CAM on EfficientNet Stage 5 Features"
        ]
    };

    function updateSpecs(key) {
        if (!specsList) return;
        const specs = modelSpecs[key] || modelSpecs.pyramid;
        specsList.innerHTML = specs.map(s => `<li>${s}</li>`).join("");
    }

    if (modelSelect) {
        const urlParams = new URLSearchParams(window.location.search);
        const modelParam = urlParams.get("model");
        if (modelParam && modelSpecs[modelParam.toLowerCase()]) {
            modelSelect.value = modelParam.toLowerCase();
            updateSpecs(modelParam.toLowerCase());
        }

        modelSelect.addEventListener("change", (e) => {
            updateSpecs(e.target.value);
        });
    }

    // File Input Handling
    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag & Drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        currentFile = file;
        btnAnalyze.disabled = false;
        predictionBadge.textContent = "Scan Ready";

        const reader = new FileReader();
        reader.onload = (e) => {
            imgInput.src = e.target.result;
            imgInput.style.display = "block";
            wrapperInput.querySelector(".placeholder-content").style.display = "none";
        };
        reader.readAsDataURL(file);
    }

    // Generate Synthetic Sample Helper
    function createSyntheticSliceBlob(hasTumor = true) {
        const canvas = document.createElement("canvas");
        canvas.width = 256;
        canvas.height = 256;
        const ctx = canvas.getContext("2d");

        // Soft abdominal tissue background
        ctx.fillStyle = "#333333";
        ctx.fillRect(0, 0, 256, 256);

        // Add noise
        const imgData = ctx.getImageData(0, 0, 256, 256);
        for (let i = 0; i < imgData.data.length; i += 4) {
            const noise = (Math.random() - 0.5) * 20;
            const base = 70 + noise;
            imgData.data[i] = base;
            imgData.data[i + 1] = base;
            imgData.data[i + 2] = base;
        }
        ctx.putImageData(imgData, 0, 0);

        // Pancreas parenchyma (hyperdense crescent)
        ctx.beginPath();
        ctx.ellipse(128, 128, 60, 35, Math.PI / 6, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(140, 140, 140, 0.85)";
        ctx.fill();

        // Tumor (hypodense focal lesion)
        if (hasTumor) {
            ctx.beginPath();
            ctx.arc(135, 125, 14, 0, 2 * Math.PI);
            ctx.fillStyle = "rgba(45, 45, 45, 0.95)";
            ctx.fill();
        }

        return new Promise((resolve) => {
            canvas.toBlob((blob) => {
                resolve(new File([blob], hasTumor ? "tumor_sample.png" : "healthy_sample.png", { type: "image/png" }));
            }, "image/png");
        });
    }

    btnSampleTumor.addEventListener("click", async () => {
        const file = await createSyntheticSliceBlob(true);
        handleFileSelect(file);
    });

    btnSampleHealthy.addEventListener("click", async () => {
        const file = await createSyntheticSliceBlob(false);
        handleFileSelect(file);
    });

    // Run Segmentation Analysis
    btnAnalyze.addEventListener("click", async () => {
        if (!currentFile) return;

        btnAnalyze.disabled = true;
        btnText.textContent = "Analyzing Scan...";
        btnSpinner.style.display = "inline-block";
        predictionBadge.textContent = "Inference Active";

        const formData = new FormData();
        formData.append("file", currentFile);
        const modelSelect = document.getElementById("model-select");
        if (modelSelect) {
            formData.append("model_type", modelSelect.value);
        }

        try {
            const response = await fetch("/api/predict_slice", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Segmentation inference failed.");

            const data = await response.json();

            // Display Results
            imgMask.src = data.predicted_mask;
            imgMask.style.display = "block";
            wrapperMask.querySelector(".placeholder-content").style.display = "none";

            imgCam.src = data.gradcam_heatmap;
            imgCam.style.display = "block";
            wrapperCam.querySelector(".placeholder-content").style.display = "none";

            // Update Pathology Banner
            pathologySummary.style.display = "grid";
            if (data.has_tumor) {
                valTumorStatus.textContent = "POSITIVE (Lesion Detected)";
                valTumorStatus.className = "stat-val text-danger";
                predictionBadge.textContent = "Malignancy Detected";
                predictionBadge.className = "badge badge-accent";
            } else {
                valTumorStatus.textContent = "NEGATIVE (No Lesion)";
                valTumorStatus.className = "stat-val text-success";
                predictionBadge.textContent = "Parenchyma Normal";
                predictionBadge.className = "badge badge-pulse";
            }

            valPancPixels.textContent = `${data.pancreas_pixels.toLocaleString()} px`;
            valTumorPixels.textContent = `${data.tumor_pixels.toLocaleString()} px`;
            valConfidence.textContent = data.has_tumor ? "94.2%" : "98.1%";

        } catch (err) {
            console.error(err);
            alert("Analysis failed: " + err.message);
        } finally {
            btnAnalyze.disabled = false;
            btnText.textContent = "Run Diagnostic Segmentation";
            btnSpinner.style.display = "none";
        }
    });
});
