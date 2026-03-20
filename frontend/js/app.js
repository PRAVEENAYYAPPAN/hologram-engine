/**
 * Hologram Engine — Main Application
 * 
 * Orchestrates the entire UI flow:
 * 1. Image upload / camera capture
 * 2. Backend API calls (detect → retrieve → generate)
 * 3. Three.js hologram rendering
 * 4. Analysis panel updates
 */

import * as API from './api.js';
import * as Camera from './camera.js';
import * as Hologram from './hologram.js';

// ─── DOM References ─────────────────────────────────────
const $ = (id) => document.getElementById(id);

const serverStatus = $('server-status');
const tabUpload = $('tab-upload');
const tabCamera = $('tab-camera');
const tabContentUpload = $('tab-content-upload');
const tabContentCamera = $('tab-content-camera');
const uploadZone = $('upload-zone');
const fileInput = $('file-input');
const previewContainer = $('preview-container');
const previewImage = $('preview-image');
const clearPreview = $('clear-preview');
const processBtn = $('process-btn');
const holoCanvas = $('hologram-canvas');
const holoPlaceholder = $('hologram-placeholder');
const processingOverlay = $('processing-overlay');

// Camera & Prompt
const cameraVideo = $('camera-video');
const cameraCanvas = $('camera-canvas');
const btnStartCamera = $('btn-start-camera');
const btnTakePhoto = $('btn-take-photo');
const tabPrompt = $('tab-prompt');
const tabContentPrompt = $('tab-content-prompt');
const textPromptInput = $('text-prompt-input');

// Hero Title
const heroTitle = document.querySelector('.viewport-hero-title');

// Control buttons
const btnFixOrientation = $('btn-fix-orientation');
const btnResetCamera = $('btn-reset-camera');
const btnWireframe = $('btn-wireframe');
const btnAutoRotate = $('btn-auto-rotate');
const btnDownload = $('btn-download');

// ─── State ──────────────────────────────────────────────
let currentImageBlob = null;
let currentTextPrompt = "";
let currentTab = 'upload';
let isProcessing = false;
let lastModelUrl = null;

// ─── Initialization ─────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    console.log("[App] DOMContentLoaded triggered.");
    
    try {
        console.log("[App] Instantiating Hologram Renderer...");
        initHologramRenderer();
        console.log("[App] Instantiating Tabs...");
        initTabs();
        console.log("[App] Instantiating Upload...");
        initUpload();
        console.log("[App] Instantiating Camera Controls...");
        initCameraControls();
        console.log("[App] Instantiating Prompt Controls...");
        initPromptControls();
        console.log("[App] Instantiating Action Controls...");
        initControls();
        console.log("[App] Instantiating Background Particles...");
        initBackgroundParticles();
        console.log("[App] Checking Server Health...");
        checkServerHealth();

        console.log("[App] Starting Camera system via camera.js...");
        Camera.init(cameraVideo, cameraCanvas);

        setInterval(checkServerHealth, 15000);
        console.log("[App] Initialization complete!");
    } catch (e) {
        console.error("[App] FATAL INITIALIZATION ERROR:", e);
    }
});

// ─── Health Check ───────────────────────────────────────
let _wasOffline = false;
async function checkServerHealth() {
    // Show "waking up" state if previously offline (Render free tier cold start)
    if (_wasOffline) {
        serverStatus.querySelector('.status-dot').style.background = '#ffaa00';
        serverStatus.querySelector('.status-dot').style.boxShadow = '0 0 10px #ffaa00';
        serverStatus.querySelector('span').textContent = 'WAKING UP...';
        serverStatus.style.color = '#ffaa00';
        serverStatus.style.borderColor = 'rgba(255, 170, 0, 0.3)';
    }

    const healthy = await API.checkHealth();
    _wasOffline = !healthy;

    serverStatus.querySelector('.status-dot').style.background = healthy ? '#00ff88' : '#ff0055';
    serverStatus.querySelector('.status-dot').style.boxShadow = healthy ? '0 0 10px #00ff88' : '0 0 10px #ff0055';
    serverStatus.querySelector('span').textContent = healthy ? 'SYS.ONLINE' : 'SYS.OFFLINE';
    serverStatus.style.color = healthy ? '#00ff88' : '#ff0055';
    serverStatus.style.borderColor = healthy ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 0, 85, 0.3)';

    // If offline, retry sooner (10s) to catch the cold start wake-up
    if (!healthy) {
        setTimeout(checkServerHealth, 10000);
    }
}

// ─── Hologram Renderer ─────────────────────────────────
function initHologramRenderer() {
    Hologram.init(holoCanvas);
}

// ─── Tab Switching ──────────────────────────────────────
function initTabs() {
    tabUpload.addEventListener('click', () => switchTab('upload'));
    tabCamera.addEventListener('click', () => switchTab('camera'));
    tabPrompt.addEventListener('click', () => switchTab('prompt'));
}

function switchTab(tab) {
    if (currentTab === 'camera' && tab !== 'camera') {
        Camera.stopCamera();
        btnStartCamera.style.display = 'block';
        btnTakePhoto.style.display = 'none';
        cameraVideo.style.display = 'none';
    }

    currentTab = tab;
    currentImageBlob = null;
    currentTextPrompt = "";
    processBtn.disabled = true;

    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    $('tab-' + tab).classList.add('active');
    $('tab-content-' + tab).classList.add('active');

    if (tab === 'prompt') {
        processBtn.disabled = textPromptInput.value.trim().length === 0;
    }
}

// ─── Upload Handling ────────────────────────────────────
function initUpload() {
    uploadZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    clearPreview.addEventListener('click', () => {
        currentImageBlob = null;
        previewContainer.style.display = 'none';
        uploadZone.style.display = 'flex';
        processBtn.disabled = true;
        fileInput.value = '';
    });

    processBtn.addEventListener('click', () => {
        if (isProcessing) return;
        if (currentTab === 'upload' || currentTab === 'camera') {
            if (currentImageBlob) runDetectionPipeline();
        } else if (currentTab === 'prompt') {
            if (currentTextPrompt) runDetectionPipeline();
        }
    });
}

function initCameraControls() {
    btnStartCamera.addEventListener('click', async () => {
        const started = await Camera.startCamera();
        if (started) {
            btnStartCamera.style.display = 'none';
            btnTakePhoto.style.display = 'block';
            cameraVideo.style.display = 'block';
        } else {
            showToast('Camera access denied or unavailable', 'error');
        }
    });

    btnTakePhoto.addEventListener('click', async () => {
        const blob = await Camera.captureFrame();
        if (blob) {
            handleFile(blob);
            switchTab('upload');
            showToast('Frame captured successfully', 'success');
        }
    });
}

function initPromptControls() {
    textPromptInput.addEventListener('input', (e) => {
        currentTextPrompt = e.target.value.trim();
        processBtn.disabled = currentTextPrompt.length === 0;
    });
    textPromptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && currentTextPrompt.length > 0) {
            processBtn.click();
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file', 'error');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showToast('Image too large (max 10MB)', 'error');
        return;
    }

    currentImageBlob = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'flex';
        uploadZone.style.display = 'none';
        processBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ─── Main Pipeline ──────────────────────────────────────
async function runDetectionPipeline() {
    if (isProcessing) return;

    isProcessing = true;
    processBtn.classList.add('loading');
    processBtn.disabled = true;
    showProcessingOverlay(true);

    try {
        let result = null;
        
        if (currentTab === 'prompt') {
            updateProcessingTask('task-detect', 'done');
            updateProcessingTask('task-semantic', 'active');
            result = await API.runPromptPipeline(currentTextPrompt);
        } else {
            updateProcessingTask('task-detect', 'active');
            result = await API.runPipeline(currentImageBlob);
        }

        if (!result.success) throw new Error('Pipeline returned unsuccessful result');

        updateProcessingTask('task-detect', 'done');
        updateProcessingTask('task-semantic', 'active');
        await sleep(300);

        updateAnalysisPanel(result);

        updateProcessingTask('task-semantic', 'done');
        updateProcessingTask('task-retrieval', 'active');
        await sleep(300);

        updateProcessingTask('task-retrieval', 'done');
        updateProcessingTask('task-hologram', 'active');

        if (result.model && result.model.url) {
            const modelUrl = API.getModelUrl(result.model.url);
            lastModelUrl = modelUrl;
            await Hologram.loadModel(modelUrl);
            holoPlaceholder.style.display = 'none';
        }

        updateProcessingTask('task-hologram', 'done');

        await sleep(500);
        showProcessingOverlay(false);
        showToast(`Hologram generated: ${result.detection.semantic_class}`, 'success');

    } catch (error) {
        console.error('[Pipeline] Error:', error);
        showProcessingOverlay(false);
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        isProcessing = false;
        processBtn.classList.remove('loading');
        processBtn.disabled = false;
    }
}

// ─── Analysis Panel Update ──────────────────────────────
function updateAnalysisPanel(result) {
    const det = result.detection;
    const model = result.model;

    const classEl = $('info-semantic-class');
    const yoloEl = $('info-yolo-label');
    
    if (classEl && yoloEl) {
        classEl.textContent = det.semantic_class || 'Unknown';
        yoloEl.textContent = model.match_name ? `RETRIEVED: ${model.match_name.toUpperCase()}` : (det.yolo_label || '').toUpperCase();
        
        // Ensure hero title is visible
        if (heroTitle) {
            heroTitle.classList.add('active');
        }
    }
}

// ─── Hologram Controls ─────────────────────────────────
function initControls() {
    btnFixOrientation?.addEventListener('click', () => {
        Hologram.fixOrientation();
        showToast('Orientation adjustments applied', 'success');
    });

    btnResetCamera.addEventListener('click', () => Hologram.resetCamera());

    btnWireframe.addEventListener('click', () => {
        const active = Hologram.toggleWireframe();
        btnWireframe.classList.toggle('active', active);
    });

    btnAutoRotate.addEventListener('click', () => {
        const active = Hologram.toggleAutoRotate();
        btnAutoRotate.classList.toggle('active', active);
    });

    // NOTE: removed direct 'click' listener from btnDownload because it's defined separately below
    // Actually, I map it here:
    btnDownload?.addEventListener('click', () => {
        if (lastModelUrl) {
            const a = document.createElement('a');
            a.href = lastModelUrl;
            a.download = 'hologram_model.glb';
            a.click();
            showToast('Downloading model...', 'info');
        } else {
            showToast('No model loaded to download', 'error');
        }
    });

    btnAutoRotate.classList.add('active');
}

// ─── Processing Overlay ─────────────────────────────────
function showProcessingOverlay(show) {
    processingOverlay.style.display = show ? 'flex' : 'none';
    if (show) {
        document.querySelectorAll('.task').forEach(t => t.className = 'task');
    }
}

function updateProcessingTask(taskId, state) {
    const task = $(taskId);
    if (!task) return;
    task.className = 'task ' + state;
    const icon = task.querySelector('i');
    if (icon) {
        if (state === 'active') icon.className = 'fas fa-circle-notch fa-spin';
        else if (state === 'done') icon.className = 'fas fa-check-circle';
        else icon.className = 'fas fa-spinner';
    }
}

// ─── Toast Notifications ────────────────────────────────
function showToast(message, type = 'info') {
    let toastContainer = $('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.style.position = 'fixed';
        toastContainer.style.bottom = '20px';
        toastContainer.style.right = '20px';
        toastContainer.style.zIndex = '9999';
        toastContainer.style.display = 'flex';
        toastContainer.style.flexDirection = 'column';
        toastContainer.style.gap = '10px';
        document.body.appendChild(toastContainer);
    }
    const toast = document.createElement('div');
    toast.style.padding = '10px 20px';
    toast.style.background = type === 'error' ? 'rgba(255,0,85,0.9)' : 'rgba(0, 240, 255, 0.9)';
    toast.style.color = type === 'error' ? '#fff' : '#000';
    toast.style.fontFamily = 'Share Tech Mono, monospace';
    toast.style.borderRadius = '2px';
    toast.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.5s';
        setTimeout(() => toast.remove(), 500);
    }, 4000);
}

// ─── Background Particles ───────────────────────────────
function initBackgroundParticles() {
    const canvas = $('bg-particles');
    const ctx = canvas.getContext('2d');
    
    let width, height;
    const particles = [];
    const particleCount = 80;

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    function createParticle() {
        return {
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            size: Math.random() * 2 + 0.5,
            alpha: Math.random() * 0.3 + 0.1,
            color: Math.random() > 0.5 ? '0, 240, 255' : '0, 136, 255',
        };
    }

    function initParticles() {
        particles.length = 0;
        for (let i = 0; i < particleCount; i++) {
            particles.push(createParticle());
        }
    }

    function draw() {
        ctx.clearRect(0, 0, width, height);

        for (const p of particles) {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = width;
            if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height;
            if (p.y > height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
            ctx.fill();
        }

        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 240, 255, ${0.05 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(draw);
    }

    resize();
    initParticles();
    draw();
    window.addEventListener('resize', resize);
}

// ─── Utilities ──────────────────────────────────────────
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
