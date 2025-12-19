export const elements = {
    status: document.getElementById("status"),
    backendDisplay: document.getElementById("backend-display"),
    fileInput: document.getElementById("upload"),
    resultContainer: document.getElementById("result-container"),
    canvasOverlay: document.getElementById("canvas-overlay"),
    imagePreview: document.getElementById("image-preview"),
    detCompilation: document.getElementById("detCom"),
    detInference: document.getElementById("detInf"),
    recCompilation: document.getElementById("recCom"),
    recInference: document.getElementById("recInf"),
    backend: document.getElementById("backend"),
    model: document.getElementById("model"),
    exampleLink: document.getElementById("example-link"),
    modelRadios: document.querySelectorAll('input[name="model-version"]'),
};

export function updateStatus(text) {
    if (elements.status) elements.status.textContent = text;
}

export function updateBackendDisplay(label) {
    if (elements.backendDisplay) elements.backendDisplay.textContent = label;
    if (elements.backend) elements.backend.textContent = ` · ${label}`;
}

export function updateModelDisplay(version) {
    if (!elements.model) return;
    if (version === "1") {
        elements.model.textContent = "PP-OCRv5";
    } else {
        elements.model.textContent = "PP-OCRv4";
    }
}

export function showImage(src) {
    if (elements.imagePreview) {
        elements.imagePreview.src = src;
        elements.imagePreview.style.display = "block";
    }
}

export function clearResults() {
    if (elements.resultContainer) elements.resultContainer.innerHTML = "";
}

export function addResult(text, prob) {
    if (!elements.resultContainer) return;
    const p = document.createElement("p");
    p.textContent = `[${prob.toFixed(2)}] ${text}`;
    elements.resultContainer.appendChild(p);
}

export function updatePerformance(type, time) {
    if (type === "detCom" && elements.detCompilation) elements.detCompilation.textContent = time.toFixed(2);
    if (type === "detInf" && elements.detInference) elements.detInference.textContent = time.toFixed(2);
    if (type === "recCom" && elements.recCompilation) elements.recCompilation.textContent = time.toFixed(2);
    if (type === "recInf" && elements.recInference) elements.recInference.textContent = time.toFixed(2);
}

export function handleUrlParams() {
    const urlParams = new URLSearchParams(window.location.search);
    const hasModelParam = urlParams.has("model");
    
    if (!hasModelParam) {
        const redirectUrl = new URL(window.location.href);
        redirectUrl.searchParams.set("model", "1");
        if (!redirectUrl.searchParams.has("device")) {
            redirectUrl.searchParams.set("device", "wasm");
        }
        window.location.replace(redirectUrl.toString());
        return null; // Redirecting
    }
    
    return {
        modelVersion: urlParams.get("model") || "0",
        device: urlParams.get("device") || "wasm",
        image: urlParams.get("image"),
        hasModelParam: hasModelParam,
        hasDeviceParam: urlParams.has("device")
    };
}

export function setupUI(callbacks) {
    const { onUpload, onExampleClick, onModelChange, modelVersion } = callbacks;
    
    // File Upload
    if (elements.fileInput) {
        elements.fileInput.addEventListener("change", async (e) => {
            if (e.target.files.length > 0) {
                await onUpload(e.target.files[0]);
            }
        });
    }

    // Example Link
    if (elements.exampleLink) {
        elements.exampleLink.addEventListener("click", async (e) => {
            e.preventDefault();
            await onExampleClick();
        });
    }

    // Model Version Selection
    elements.modelRadios.forEach((radio) => {
        radio.checked = radio.value === modelVersion;
        radio.addEventListener("change", (e) => {
            onModelChange(e.target.value);
        });
    });
    
    updateModelDisplay(modelVersion);
}
