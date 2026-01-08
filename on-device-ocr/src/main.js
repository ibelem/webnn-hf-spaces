import * as ort from "onnxruntime-web/webgpu";
ort.env.wasm.wasmPaths = "/onnxruntime-web-1-23-2-dists/";
import { getHuggingFaceDomain } from "./utils.js";
import * as ui from "./ui.js";

import { 
    loadImage, 
    preprocessDetection, 
    postprocessDetection, 
    splitIntoLineImages, 
    preprocessRecognition, 
    decodeText,
    drawBoxes,
    groupLines
} from "./ocr-processing.js";

ort.env.logLevel = "verbose";

// Configuration
const hfDomain = await getHuggingFaceDomain();

// Handle URL Params
const params = ui.handleUrlParams();
if (!params) {
    // Redirecting...
    throw new Error("Redirecting...");
}

const { modelVersion, device: currentBackend, image: imageParam } = params;

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
let totalRecInferenceTime = null;

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

// Initialize
async function init() {
    try {
        ui.updateStatus("Loading OpenCV...");

        // Wait for OpenCV with timeout
        const openCVTimeout = 30000; // 30 seconds
        const startTime = Date.now();
        while (typeof cv === "undefined") {
            if (Date.now() - startTime > openCVTimeout) {
                throw new Error("OpenCV loading timeout. Please refresh the page.");
            }
            await new Promise(r => setTimeout(r, 100));
        }

        ui.updateStatus("OpenCV Loaded. Loading Dictionary...");

        // Load Dictionary with retry logic
        let retries = 3;
        let lastError;
        while (retries > 0) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

                const response = await fetch(MODELS.dic, {
                    signal: controller.signal,
                    cache: "default"
                });
                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`Failed to load dictionary: ${response.status} ${response.statusText}`);
                }
                
                const text = await response.text();
                dictionary = text.split("\n");
                dictionary.push(" ");
                break; // Success, exit retry loop
            } catch (error) {
                lastError = error;
                retries--;
                console.error(`Dictionary loading attempt failed (${3 - retries}/3):`, error);
                if (retries > 0) {
                    ui.updateStatus(`Retrying dictionary load... (${3 - retries}/3)`);
                    await new Promise(r => setTimeout(r, 1000));
                }
            }
        }
        
        if (dictionary.length === 0) {
            throw lastError || new Error("Failed to load dictionary after 3 attempts. Please check your network connection.");
        }

        ui.updateBackendDisplay(currentBackendLabel);

        const shouldAutoRun = imageParam && params.hasModelParam && params.hasDeviceParam;

        if (shouldAutoRun) {
            ui.updateStatus("Ready. Processing URL image...");
        } else {
            ui.updateStatus("Ready. Upload image.");
        }

        ui.setupUI({
            onUpload: async (file) => {
                await runOCR(file);
            },
            onExampleClick: async () => {
                await runOCR("https://ibelem.github.io/webnn-hf-spaces/on-device-ocr/assets/invoice.jpg");
            },
            onModelChange: (newVersion) => {
                const url = new URL(window.location.href);
                url.searchParams.set("model", newVersion);
                window.location.href = url.toString();
            },
            modelVersion: modelVersion
        });

        if (shouldAutoRun) {
            await runOCR(imageParam);
        }
    } catch (error) {
        console.error("Initialization error:", error);
        ui.updateStatus(`Error: ${error.message}. Please check your network connection and refresh.`);
        // Show device links so user can try different backend
        const deviceLinks = document.querySelector('.device-links');
        if (deviceLinks) deviceLinks.style.display = 'block';
    }
}

async function getSession(type) {
    if (type === "det" && detSession) return detSession;
    if (type === "rec" && recSession) return recSession;
    
    ui.updateStatus(`Loading ${type} model (${currentBackend})...`);
    
    const options = {
        executionProviders: [currentBackend]
    };

    options.logSeverityLevel = 0;
    options.logVerbosityLevel = 0;
    
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
        ui.updatePerformance("detCom", compilationTime);
        console.log(`Detection model compilation time: ${compilationTime.toFixed(2)}ms`);
    } else {
        ui.updatePerformance("recCom", compilationTime);
        console.log(`Recognition model compilation time: ${compilationTime.toFixed(2)}ms`);
    }
    
    if (type === "det" && detSession === null) detSession = session;
    else if (type === "rec" && recSession === null) recSession = session;
    
    return session;
}

async function runOCR(file) {
    try {
        ui.updateStatus("Processing image...");
        ui.clearResults();
        
        // Load Image
        const image = await loadImage(file);
        ui.showImage(image.src);
        
        // 1. Detection
        const detSess = await getSession("det");
        
        ui.updateStatus("Running Detection...");
        
        const { tensor: detInput, width: detW, height: detH, originalWidth, originalHeight, imageData } = preprocessDetection(image);
        const detFeeds = {};
        detFeeds[detSess.inputNames[0]] = detInput;

        const detStart = performance.now();
        const detOutput = await detSess.run(detFeeds);
        const detEnd = performance.now();
        const detInferenceTime = detEnd - detStart;
        ui.updatePerformance("detInf", detInferenceTime);
        console.log(`Detection inference time: ${detInferenceTime.toFixed(2)}ms`);
        const detResult = detOutput[detSess.outputNames[0]];
        const maskImageData = postprocessDetection(detResult, detW, detH);
        
        
        // 2. Split into lines
        ui.updateStatus("Splitting lines...");
        
        const lineImages = splitIntoLineImages(maskImageData, image);
        
        // Draw boxes on overlay
        drawBoxes(ui.elements.canvasOverlay, originalWidth, originalHeight, lineImages);
        
        // 3. Recognition
        ui.updateStatus("Running Recognition...");
        const recSess = await getSession("rec");
        
        let fullText = "";
        totalRecInferenceTime = 0;
        
        const results = [];
        let acceptedCount = 0;
        let skippedCount = 0;
        for (let i = 0; i < lineImages.length; i++) {
            const line = lineImages[i];
            const recInput = preprocessRecognition(line.mat);
            
            const recFeeds = {};
            recFeeds[recSess.inputNames[0]] = recInput;
            
            const recStart = performance.now();
            const recOutput = await recSess.run(recFeeds);
            const recEnd = performance.now();
            const recInferenceTime = recEnd - recStart;
            console.log(`[line ${line.id}] Recognition inference time: ${recInferenceTime.toFixed(2)}ms`);
            totalRecInferenceTime += recInferenceTime;
            const recResult = recOutput[recSess.outputNames[0]];
            
            const { text, meanProb } = decodeText(recResult, dictionary);
            
            // meanProb is a mean logit (reference-style), not a probability.
            // Reference filters around mean >= 0.5.
            if (text !== "" && meanProb >= 0.5) {
                acceptedCount++;
                results.push({
                    text: text,
                    mean: meanProb,
                    box: line.box
                });
                console.log(`[line ${line.id}]: [${meanProb.toFixed(4)}] ${text}`);
            } else {
                skippedCount++;
                console.log(`[line ${line.id}] Low confidence line skipped: "${text}" (${meanProb.toFixed(4)})`);
            }
            
            // Clean up Mat
            line.mat.delete();
        }

        console.log(`[rec summary] total=${lineImages.length} accepted(mean>=0.5)=${acceptedCount} skipped=${skippedCount}`);

        const groupedResults = groupLines(results);
        groupedResults.forEach(res => {
            fullText += res.text + "\n";
            ui.addResult(res.text, res.mean);
        });

        ui.updatePerformance("recInf", totalRecInferenceTime);
        
        ui.updateStatus(`${currentBackendLabel} OCR complete.`);
        console.log("OCR Result:\n", fullText);
        
    } catch (e) {
        console.error(e);
        ui.updateStatus(`Error: ${e.message}`);
    }
}

init();
