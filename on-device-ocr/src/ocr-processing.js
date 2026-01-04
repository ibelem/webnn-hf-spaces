import * as ort from "onnxruntime-web/webgpu";
ort.env.wasm.wasmPaths = "/onnxruntime-web-1-23-2-dists/";

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
    
    // Normalize: reference uses defaults mean=[0,0,0], std=[1,1,1] -> just scale to 0-1.
    const R = [], G = [], B = [];
    for (let i = 0; i < data.length; i += 4) {
        R.push(data[i] / 255);
        G.push(data[i + 1] / 255);
        B.push(data[i + 2] / 255);
    }

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
export function postprocessDetection(output, width, height, threshold = 0.03) {
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
        
        // Clip points to image boundaries to avoid sampling outside
        scaledPoints.forEach(p => {
            p.x = Math.max(0, Math.min(p.x, originalImage.width));
            p.y = Math.max(0, Math.min(p.y, originalImage.height));
        });

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
                 id: i,
                 mat: dst_rot,
                 box: orderedPoints // Keep box for visualization
             });
        } else {
            lineImages.push({
                id: i,
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
    // Normalize to 0-1 (reference defaults mean=[0,0,0], std=[1,1,1])
    const data = resized.data; // RGBA
    const R = [], G = [], B = [];
    
    for (let i = 0; i < data.length; i += 4) {
        R.push(data[i] / 255);
        G.push(data[i + 1] / 255);
        B.push(data[i + 2] / 255);
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
    
    // Reference implementation uses raw max logit per timestep (no softmax).
    // That means meanProb is a mean logit, not a 0-1 probability.
    for (let i = 0; i < seq; i++) {
        const offset = i * classes;

        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let j = 0; j < classes; j++) {
            const v = data[offset + j];
            if (v > maxVal) {
                maxVal = v;
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

// Draw bounding boxes on canvas
export function drawBoxes(canvas, width, height, boxes) {
    // Use displayed canvas size (set via CSS) to align with the scaled image preview
    const displayW = canvas.clientWidth || width;
    const displayH = canvas.clientHeight || height;
    const sx = displayW / width;
    const sy = displayH / height;

    canvas.width = displayW;
    canvas.height = displayH;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, displayW, displayH);
    
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'red';
    
    boxes.forEach(line => {
        const box = line.box; // [TL, TR, BR, BL]
        ctx.beginPath();
        ctx.moveTo(box[0].x * sx, box[0].y * sy);
        ctx.lineTo(box[1].x * sx, box[1].y * sy);
        ctx.lineTo(box[2].x * sx, box[2].y * sy);
        ctx.lineTo(box[3].x * sx, box[3].y * sy);
        ctx.closePath();
        ctx.stroke();

        // Draw line id for debugging which box recognized correctly
        const cx = box[0].x * sx - 15;
        const cy = (box[0].y + box[2].y) * 0.5 * sy + 3;
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.font = '10px';
        ctx.strokeText(String(line.id), cx, cy);
        ctx.fillStyle = 'red';
        ctx.fillText(String(line.id), cx, cy);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 1;
    });
}

export function groupLines(results) {
    if (results.length === 0) return [];

    function calculateAverageHeight(results) {
        let totalHeight = 0;
        for (const res of results) {
            const box = res.box; 
            const h = Math.abs(box[2].y - box[0].y);
            totalHeight += h;
        }
        return totalHeight / results.length;
    }

    const averageHeight = calculateAverageHeight(results);
    const groups = [];

    for (const res of results) {
        const box = res.box;
        const midline = (box[0].y + box[2].y) / 2;
        
        const group = groups.find(g => {
            const gBox = g[0].box;
            const gMidline = (gBox[0].y + gBox[2].y) / 2;
            return Math.abs(gMidline - midline) < averageHeight / 2;
        });
        
        if (group) {
            group.push(res);
        } else {
            groups.push([res]);
        }
    }

    for (const group of groups) {
        group.sort((a, b) => a.box[0].x - b.box[0].x);
    }

    groups.sort((a, b) => a[0].box[0].y - b[0].box[0].y);

    return groups.map(group => {
        const texts = group.map(item => item.text);
        const means = group.map(item => item.mean);
        const mean = means.reduce((a, b) => a + b, 0) / means.length;
        
        return {
            text: texts.join(" "),
            mean: mean
        };
    });
}
