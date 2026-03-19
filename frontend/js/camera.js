/**
 * Camera Module
 * 
 * Handles webcam access, video streaming, and frame capture.
 */

let stream = null;
let videoEl = null;
let canvasEl = null;

/**
 * Initialize camera with the given video and canvas elements.
 */
export function init(video, canvas) {
    videoEl = video;
    canvasEl = canvas;
}

/**
 * Start the camera stream.
 * @returns {Promise<boolean>} true if started successfully
 */
export async function startCamera() {
    if (stream) return true;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
            audio: false,
        });
        
        if (videoEl) {
            videoEl.srcObject = stream;
            await videoEl.play();
        }
        
        return true;
    } catch (err) {
        console.error('[Camera] Failed to start:', err);
        return false;
    }
}

/**
 * Stop the camera stream.
 */
export function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (videoEl) {
        videoEl.srcObject = null;
    }
}

/**
 * Capture the current video frame as an image Blob.
 * @returns {Promise<Blob|null>} JPEG blob or null
 */
export async function captureFrame() {
    if (!videoEl || !canvasEl || !stream) return null;

    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;

    const ctx = canvasEl.getContext('2d');
    ctx.drawImage(videoEl, 0, 0);

    return new Promise(resolve => {
        canvasEl.toBlob(blob => resolve(blob), 'image/jpeg', 0.92);
    });
}

/**
 * Check if camera is currently active.
 * @returns {boolean}
 */
export function isActive() {
    return stream !== null;
}
