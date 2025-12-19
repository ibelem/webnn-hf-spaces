import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.webgpu.min.js";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
import { getHuggingFaceDomain } from "./utils.js";

import { 
    loadImage, 
    preprocessDetection, 
    postprocessDetection, 
    splitIntoLineImages, 
    preprocessRecognition, 
    decodeText 
} from "./ocr-processing.js";

// Configuration
const hfDomain = await getHuggingFaceDomain();
const urlParams = new URLSearchParams(window.location.search);
const modelVersion = urlParams.get("model") || "0"; // 0 for v4, 1 for v5

let baseUrl, MODELS;

if (modelVersion === "1") {
    // v5
    baseUrl = `https://${hfDomain}/webnn/PP-OCRv5-ONNX/resolve/main`;
    MODELS = {
        det: `${baseUrl}/ch_PP-OCRv5_det.onnx`,
        rec: `${baseUrl}/ch_PP-OCRv5_rec.onnx`,
        dic: `${baseUrl}/ch_PP-OCRv5_dict.txt`
    };
} else {
    // v4 (default)
    baseUrl = `https://${hfDomain}/webnn/PP-OCRv4-ONNX/resolve/main`;
    MODELS = {
        det: `${baseUrl}/ch_PP-OCRv4_det.onnx`,
        rec: `${baseUrl}/ch_PP-OCRv4_rec.onnx`,
        dic: `${baseUrl}/ch_PP-OCR_keys_v1.txt`
    };
}

// State
let detSession = null;
let recSession = null;
let dictionary = [];
let detCompilationTime = null;
let detInferenceTime = null;
let recCompilationTime = null;
let recInferenceTime = null;
let totalRecInferenceTime = null;

// urlParams already declared above
let currentBackend = urlParams.get("device") || "wasm";
let currentBackendLabel = "";
if (currentBackend === "wasm") {
    currentBackendLabel = "Wasm";
} else if (currentBackend === "webgpu") {
    currentBackendLabel = "WebGPU";
} else if (currentBackend.startsWith("webnn")) {
    const deviceType = currentBackend.includes("-") ? currentBackend.split("-")[1] : "GPU";
    currentBackendLabel = `WebNN (${deviceType.toUpperCase()})`;
} else {
    currentBackendLabel = currentBackend;
}

// UI Elements
const statusEl = document.getElementById("status");
const backendDisplay = document.getElementById("backend-display");
const fileInput = document.getElementById("upload");
const resultContainer = document.getElementById("result-container");
const canvasOverlay = document.getElementById("canvas-overlay");
const imagePreview = document.getElementById("image-preview");
const detCompilationEl = document.getElementById("detCom");
const detInferenceEl = document.getElementById("detInf");
const recCompilationEl = document.getElementById("recCom");
const recInferenceEl = document.getElementById("recInf");
const backendEl = document.getElementById("backend");

// Initialize
async function init() {
    statusEl.textContent = "Loading OpenCV...";
    // Wait for OpenCV
    while (typeof cv === "undefined") {
        await new Promise(r => setTimeout(r, 100));
    }
    statusEl.textContent = "OpenCV Loaded. Loading Dictionary...";
    
    // Load Dictionary
    const response = await fetch(MODELS.dic);
    const text = await response.text();
    dictionary = text.split("\n");
    // Add space at the end if needed, or handle it in decode
    dictionary.push(" "); 
    
    backendDisplay.textContent = currentBackendLabel;
    backendEl.textContent = ` · ${currentBackendLabel}`;
    statusEl.textContent = "Ready. Upload image.";
    
    // File Upload
    fileInput.addEventListener("change", async (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            await runOCR(file);
        }
    });

    // Example Link
    const exampleLink = document.getElementById("example-link");
    if (exampleLink) {
        exampleLink.addEventListener("click", async (e) => {
            e.preventDefault();
            await runOCR("img/invoice.jpg");
        });
    }

    // Model Version Selection
    const modelRadios = document.getElementsByName("model-version");
    for (const radio of modelRadios) {
        if (radio.value === modelVersion) {
            radio.checked = true;
        }
        radio.addEventListener("change", (e) => {
            const newVersion = e.target.value;
            const url = new URL(window.location);
            url.searchParams.set("model", newVersion);
            window.location.href = url.toString();
        });
    }
}

async function getSession(type) {
    if (type === "det" && detSession) return detSession;
    if (type === "rec" && recSession) return recSession;
    
    statusEl.textContent = `Loading ${type} model (${currentBackend})...`;
    
    const options = {
        executionProviders: [currentBackend]
    };
    
    if (currentBackend.startsWith("webnn")) {
        const deviceType = currentBackend.includes("-") ? currentBackend.split("-")[1] : "gpu";
        options.executionProviders = [{
            name: "webnn",
            deviceType,
            powerPreference: "default"
        }];
    } else if (currentBackend === "webgpu") {
        options.executionProviders = [{
            name: currentBackend,
            deviceType: "gpu",
            powerPreference: "default"
        }];
    } else {
        options.executionProviders = [{
            name: currentBackend,
            deviceType: "cpu",
            powerPreference: "default"
        }];
    }
    
    const path = type === "det" ? MODELS.det : MODELS.rec;
    let startTime = performance.now();
    const session = await ort.InferenceSession.create(path, options);
    let endTime = performance.now();
    const compilationTime = endTime - startTime;
    
    if (type === "det") {
        detCompilationTime = compilationTime;
        detCompilationEl.textContent = `${detCompilationTime.toFixed(2)}`;
        console.log(`Detection model compilation time: ${compilationTime.toFixed(2)}ms`);
    } else {
        recCompilationTime = compilationTime;
        recCompilationEl.textContent = `${recCompilationTime.toFixed(2)}`;
        console.log(`Recognition model compilation time: ${compilationTime.toFixed(2)}ms`);
    }
    
    if (type === "det" && detSession === null) detSession = session;
    else if (type === "rec" && recSession === null) recSession = session;
    
    return session;
}

async function runOCR(file) {
    try {
        statusEl.textContent = "Processing image...";
        resultContainer.innerHTML = "";
        
        // Load Image
        const image = await loadImage(file);
        imagePreview.src = image.src;
        imagePreview.style.display = "block";
        
        // 1. Detection
        const detSess = await getSession("det");
        
        statusEl.textContent = "Running Detection...";
        
        const { tensor: detInput, width: detW, height: detH, originalWidth, originalHeight, imageData } = preprocessDetection(image);
        const detFeeds = {};
        detFeeds[detSess.inputNames[0]] = detInput;

        const detStart = performance.now();
        const detOutput = await detSess.run(detFeeds);
        const detEnd = performance.now();
        detInferenceTime = detEnd - detStart;
        detInferenceEl.textContent = `${detInferenceTime.toFixed(2)}`;
        console.log(`Detection inference time: ${detInferenceTime.toFixed(2)}ms`);
        const detResult = detOutput[detSess.outputNames[0]];
        const maskImageData = postprocessDetection(detResult, detW, detH);
        
        
        // 2. Split into lines
        statusEl.textContent = "Splitting lines...";
        
        const lineImages = splitIntoLineImages(maskImageData, image);
        
        // Draw boxes on overlay
        canvasOverlay.width = originalWidth;
        canvasOverlay.height = originalHeight;
        const ctx = canvasOverlay.getContext("2d");
        ctx.clearRect(0, 0, originalWidth, originalHeight);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        
        lineImages.forEach(line => {
            const box = line.box; // TL, TR, BR, BL
            ctx.beginPath();
            ctx.moveTo(box[0].x, box[0].y);
            ctx.lineTo(box[1].x, box[1].y);
            ctx.lineTo(box[2].x, box[2].y);
            ctx.lineTo(box[3].x, box[3].y);
            ctx.closePath();
            ctx.stroke();
        });
        
        // 3. Recognition
        statusEl.textContent = "Running Recognition...";
        const recSess = await getSession("rec");
        
        let fullText = "";
        
        for (let i = 0; i < lineImages.length; i++) {
            const line = lineImages[i];
            const recInput = preprocessRecognition(line.mat);
            
            const recFeeds = {};
            recFeeds[recSess.inputNames[0]] = recInput;
            
            const recStart = performance.now();
            const recOutput = await recSess.run(recFeeds);
            const recEnd = performance.now();
            recInferenceTime = recEnd - recStart;
            console.log(`Recognition inference time (line ${i + 1}): ${recInferenceTime.toFixed(2)}ms`);
            totalRecInferenceTime = (totalRecInferenceTime || 0) + recInferenceTime;
            const recResult = recOutput[recSess.outputNames[0]];
            
            const { text, meanProb } = decodeText(recResult, dictionary);
            
            if (meanProb > 0.3) { // Confidence threshold
                fullText += text + "\n";
                
                // Display result
                const p = document.createElement("p");
                p.textContent = `[${meanProb.toFixed(2)}] ${text}`;
                resultContainer.appendChild(p);
            } else {
                console.log(`Low confidence line skipped: "${text}" (${meanProb.toFixed(2)})`);
            }
            
            // Clean up Mat
            line.mat.delete();
        }

        recInferenceEl.textContent = `${totalRecInferenceTime.toFixed(2)}`;
        
        statusEl.textContent = `${currentBackendLabel} OCR complete.`;
        console.log("OCR Result:\n", fullText);
        
    } catch (e) {
        console.error(e);
        statusEl.textContent = `Error: ${e.message}`;
    }
}

init();
