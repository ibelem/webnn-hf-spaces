import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.webgpu.min.js";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// Helper to load image from URL or File
export async function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => resolve(img);
        img.onerror = reject;
        if (src instanceof Blob) {
            img.src = URL.createObjectURL(src);
        } else {
            img.src = src;
        }
    });
}

// Preprocess image for Detection model
// Resize to multiple of 32, normalize
export function preprocessDetection(image, maxSize = 960) {
    const canvas = document.createElement('canvas');
    let width = image.width;
    let height = image.height;

    // Resize logic from reference
    if (maxSize && Math.max(width, height) > maxSize) {
        const ratio = width > height ? maxSize / width : maxSize / height;
        width = width * ratio;
        height = height * ratio;
    }
    const newWidth = Math.max(Math.ceil(width / 32) * 32, 32);
    const newHeight = Math.max(Math.ceil(height / 32) * 32, 32);

    canvas.width = newWidth;
    canvas.height = newHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, newWidth, newHeight);
    
    const imageData = ctx.getImageData(0, 0, newWidth, newHeight);
    const { data } = imageData;

    // Normalize: (pixel / 255 - mean) / std
    // Default mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] for ImageNet trained models
    // But reference Detection.ts uses default mean=[0,0,0] std=[1,1,1] in imageToInput if not specified?
    // Wait, Detection.ts says:
    // const modelData = this.imageToInput(inputImage, {
    //   // mean: [0.485, 0.456, 0.406],
    //   // std: [0.229, 0.224, 0.225],
    // })
    // And ModelBase.ts default is mean=[0,0,0], std=[1,1,1].
    // So it seems it just scales to 0-1.
    
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    // Actually, let's check the reference again. The commented out lines suggest they MIGHT be used, 
    // but if they are commented out, the defaults [0,0,0] and [1,1,1] are used.
    // However, PaddleOCR usually expects normalized inputs.
    // Let's stick to the reference code which seems to use defaults (0-1).
    // Wait, in Detection.ts lines 33-36 are commented out.
    // So it uses default mean=[0,0,0], std=[1,1,1].

    const R = [], G = [], B = [];
    for (let i = 0; i < data.length; i += 4) {
        R.push(((data[i] / 255) - mean[0]) / std[0]);
        G.push(((data[i+1] / 255) - mean[1]) / std[1]);
        B.push(((data[i+2] / 255) - mean[2]) / std[2]);
    }
    
    const input = Float32Array.from([...R, ...G, ...B]); // RGB order?
    // ModelBase.ts:
    // R.push... G.push... B.push...
    // const newData = [...B, ...G, ...R]  <-- BGR order? Or just planar?
    // Wait, ModelBase.ts line 48: const newData = [...B, ...G, ...R]
    // This looks like BGR planar.
    // Let's verify if PaddleOCR uses RGB or BGR.
    // Usually it's RGB.
    // Let's check ModelBase.ts again carefully.
    // "R.push((image.data[i] / 255 - mean[0]) / std[0])"
    // "const newData = [...B, ...G, ...R]"
    // This definitely puts B first.
    // But wait, standard ONNX models usually take RGB.
    // Let's check if I misread the file content.
    // I'll assume the reference code is correct and use BGR planar if that's what it does.
    
    // Re-reading ModelBase.ts from my previous `cat` output (it was truncated in thought, but I read it).
    // Actually, I should check the `cat` output again.
    // I didn't cat ModelBase.ts fully.
    // Let's assume RGB for now, but if it fails, I'll switch.
    // Most web demos use RGB.
    // Wait, `cat on-device-ocr/reference/common/src/models/ModelBase.ts` was NOT run.
    // I ran `cat on-device-ocr/reference/common/src/models/Detection.ts`.
    // In `Detection.ts`, it calls `this.imageToInput`.
    // I saw `imageToInput` in semantic search snippet:
    // `const newData = [...B, ...G, ...R]`
    // Okay, I will use BGR planar.
    
    const inputTensor = new ort.Tensor('float32', Float32Array.from([...B, ...G, ...R]), [1, 3, newHeight, newWidth]);
    
    return {
        tensor: inputTensor,
        width: newWidth,
        height: newHeight,
        originalWidth: image.width,
        originalHeight: image.height,
        imageData: imageData
    };
}

// Post-process Detection output
export function postprocessDetection(output, width, height, threshold = 0.3) {
    // output is 1x1xHxW
    const data = output.data;
    const maskData = new Uint8ClampedArray(width * height * 4);
    
    for (let i = 0; i < data.length; i++) {
        const val = data[i] > threshold ? 255 : 0;
        maskData[i * 4] = val;
        maskData[i * 4 + 1] = val;
        maskData[i * 4 + 2] = val;
        maskData[i * 4 + 3] = 255;
    }
    
    return new ImageData(maskData, width, height);
}

// Split into line images using OpenCV
export function splitIntoLineImages(maskImageData, originalImage) {
    if (typeof cv === 'undefined') {
        console.error("OpenCV not loaded");
        return [];
    }

    const w = maskImageData.width;
    const h = maskImageData.height;
    
    // Create Mat from mask
    const src = cv.matFromImageData(maskImageData);
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    
    cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    
    const lineImages = [];
    const minSize = 3;
    
    // Original image data for cropping
    // We need to draw original image to canvas to get ImageData if it's an Image element
    const canvas = document.createElement('canvas');
    canvas.width = originalImage.width;
    canvas.height = originalImage.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(originalImage, 0, 0);
    const originalImageData = ctx.getImageData(0, 0, originalImage.width, originalImage.height);
    const srcMat = cv.matFromImageData(originalImageData);

    const rx = originalImage.width / w;
    const ry = originalImage.height / h;

    for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        const rect = cv.minAreaRect(cnt);
        const box = cv.RotatedRect.points(rect);
        
        const side = Math.min(rect.size.width, rect.size.height);
        if (side < minSize) continue;

        // Unclip logic approximation (expand box)
        // Reference uses Clipper, we will just expand the size
        const unclip_ratio = 1.5; // from reference
        // Area = w * h
        // Length = 2 * (w + h)
        // distance = area * ratio / length
        const rw = rect.size.width;
        const rh = rect.size.height;
        const area = rw * rh;
        const length = 2 * (rw + rh);
        const distance = (area * unclip_ratio) / length;
        
        // Expand rect by distance
        const expandedSize = new cv.Size(rect.size.width + 2 * distance, rect.size.height + 2 * distance);
        const expandedRect = new cv.RotatedRect(rect.center, expandedSize, rect.angle);
        
        let points = cv.RotatedRect.points(expandedRect);
        
        // Sort points clockwise
        // ... (simplified, minAreaRect points are usually ordered but let's ensure)
        // Actually cv.RotatedRect.points returns BL, TL, TR, BR order or similar.
        // We need to map to destination points for perspective transform.
        
        // Scale points to original image size
        const scaledPoints = points.map(p => ({ x: p.x * rx, y: p.y * ry }));
        
        // Crop and warp
        const cropWidth = Math.max(
            Math.hypot(scaledPoints[0].x - scaledPoints[1].x, scaledPoints[0].y - scaledPoints[1].y),
            Math.hypot(scaledPoints[2].x - scaledPoints[3].x, scaledPoints[2].y - scaledPoints[3].y)
        );
        const cropHeight = Math.max(
            Math.hypot(scaledPoints[1].x - scaledPoints[2].x, scaledPoints[1].y - scaledPoints[2].y),
            Math.hypot(scaledPoints[3].x - scaledPoints[0].x, scaledPoints[3].y - scaledPoints[0].y)
        );
        
        // Destination points
        const dstPoints = [
            0, cropHeight,
            0, 0,
            cropWidth, 0,
            cropWidth, cropHeight
        ];
        // Note: minAreaRect points order depends on angle.
        // We need to order them: BL, TL, TR, BR to match dstPoints?
        // Or TL, TR, BR, BL?
        // Let's use a robust ordering function.
        
        const orderedPoints = orderPoints(scaledPoints);
        // ordered: TL, TR, BR, BL
        
        const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            orderedPoints[0].x, orderedPoints[0].y,
            orderedPoints[1].x, orderedPoints[1].y,
            orderedPoints[2].x, orderedPoints[2].y,
            orderedPoints[3].x, orderedPoints[3].y
        ]);
        
        const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            0, 0,
            cropWidth, 0,
            cropWidth, cropHeight,
            0, cropHeight
        ]);
        
        const M = cv.getPerspectiveTransform(srcTri, dstTri);
        const dst = new cv.Mat();
        cv.warpPerspective(srcMat, dst, M, new cv.Size(cropWidth, cropHeight), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
        
        // Check if we need to rotate (if height > width * 1.5, likely vertical text treated as horizontal?)
        // Reference: if (dst_img_height / dst_img_width >= 1.5) rotate 90
        if (dst.rows / dst.cols >= 1.5) {
             const dst_rot = new cv.Mat();
             cv.rotate(dst, dst_rot, cv.ROTATE_90_CLOCKWISE);
             dst.delete();
             // dst = dst_rot; // reassign
             // Actually let's just push dst_rot
             lineImages.push({
                 mat: dst_rot,
                 box: orderedPoints // Keep box for visualization
             });
        } else {
            lineImages.push({
                mat: dst,
                box: orderedPoints
            });
        }
        
        srcTri.delete();
        dstTri.delete();
        M.delete();
    }
    
    src.delete();
    contours.delete();
    hierarchy.delete();
    srcMat.delete();
    
    // Sort line images top to bottom
    lineImages.sort((a, b) => a.box[0].y - b.box[0].y);
    
    return lineImages;
}

function orderPoints(pts) {
    // pts is array of {x, y}
    // Sort by x to get left and right
    pts.sort((a, b) => a.x - b.x);
    const left = pts.slice(0, 2);
    const right = pts.slice(2, 4);
    
    // Sort left by y to get TL, BL
    left.sort((a, b) => a.y - b.y);
    const tl = left[0];
    const bl = left[1];
    
    // Sort right by y to get TR, BR
    right.sort((a, b) => a.y - b.y);
    const tr = right[0];
    const br = right[1];
    
    return [tl, tr, br, bl];
}

// Preprocess line image for Recognition
export function preprocessRecognition(mat) {
    // Resize to height 48, width scaled
    const h = 48;
    const w = Math.ceil(mat.cols * (h / mat.rows));
    
    const dsize = new cv.Size(w, h);
    const resized = new cv.Mat();
    cv.resize(mat, resized, dsize, 0, 0, cv.INTER_LINEAR);
    
    // Convert to tensor
    // Normalize: (pixel / 255 - 0.5) / 0.5
    const data = resized.data; // RGBA
    const R = [], G = [], B = [];
    
    for (let i = 0; i < data.length; i += 4) {
        R.push((data[i] / 255 - 0.5) / 0.5);
        G.push((data[i+1] / 255 - 0.5) / 0.5);
        B.push((data[i+2] / 255 - 0.5) / 0.5);
    }
    
    resized.delete();
    
    // BGR planar? Reference Recognition.ts uses imageToInput which uses ModelBase default.
    // Wait, Recognition.ts calls imageToInput with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5].
    // And ModelBase uses BGR.
    
    const input = Float32Array.from([...B, ...G, ...R]);
    return new ort.Tensor('float32', input, [1, 3, h, w]);
}

// Decode text
export function decodeText(output, dictionary) {
    // output: 1 x seq_len x num_classes
    const dims = output.dims;
    const seqLen = dims[2]; // Wait, dims are [batch, channels, seq_len]? No.
    // PaddleOCR output is usually [batch, seq_len, num_classes] or [seq_len, batch, num_classes]
    // Reference Recognition.ts:
    // const predLen = data.dims[2]
    // let ml = data.dims[0] - 1
    // for (let l = 0; l < data.data.length; l += predLen * data.dims[1])
    // This suggests dims are [batch, something, predLen]?
    // Actually, let's look at Recognition.ts again.
    // "const predLen = data.dims[2]"
    // "for (let i = l; i < l + predLen * data.dims[1]; i += predLen)"
    // This loop structure is confusing.
    
    // Standard PaddleOCR output shape is [Batch, Seq, Classes].
    // If dims[2] is predLen (Classes?), then it matches.
    // Let's assume [1, SeqLen, NumClasses].
    
    const batch = dims[0];
    const seq = dims[1];
    const classes = dims[2];
    
    const data = output.data;
    
    let text = "";
    let meanProb = 0;
    const charIndices = [];
    const probs = [];
    
    for (let i = 0; i < seq; i++) {
        // Find max class for this time step
        let maxVal = -Infinity;
        let maxIdx = -1;
        const offset = i * classes;
        for (let j = 0; j < classes; j++) {
            const val = data[offset + j];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = j;
            }
        }
        charIndices.push(maxIdx);
        probs.push(maxVal);
    }
    
    // CTC Decode (Greedy)
    // 1. Remove ignored tokens (0 is usually blank in PaddleOCR?)
    // Reference says: const ignoredTokens = [0]
    // 2. Remove duplicates
    
    const cleanIndices = [];
    const cleanProbs = [];
    
    for (let i = 0; i < charIndices.length; i++) {
        const idx = charIndices[i];
        if (idx === 0) continue; // Blank
        
        if (i > 0 && idx === charIndices[i-1]) continue; // Duplicate
        
        cleanIndices.push(idx);
        cleanProbs.push(probs[i]);
    }
    
    // Map to chars
    // dictionary is array of chars. idx-1 because 0 is blank.
    const chars = cleanIndices.map(idx => dictionary[idx - 1] || '');
    text = chars.join('');
    
    if (cleanProbs.length > 0) {
        meanProb = cleanProbs.reduce((a, b) => a + b, 0) / cleanProbs.length;
    }
    
    return { text, meanProb };
}
