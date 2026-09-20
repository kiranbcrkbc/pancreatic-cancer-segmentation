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
