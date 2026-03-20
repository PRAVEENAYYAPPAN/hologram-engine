/**
 * API Client Module
 * 
 * Handles all communication with the Hologram Engine backend API.
 */

// Backend API base URL — production Render backend
const API_BASE = window.HOLOGRAM_API_BASE || 'https://hologram-engine.onrender.com';

/**
 * Check if the backend server is reachable.
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(5000),
        });
        const data = await response.json();
        return data.status === 'ok';
    } catch {
        return false;
    }
}

/**
 * Run the full detection + retrieval + generation pipeline.
 * 
 * @param {File|Blob} imageFile - Image to process
 * @param {function} onProgress - Progress callback (stage: string)
 * @returns {Promise<Object>} Pipeline result
 */
export async function runPipeline(imageFile, onProgress = () => {}) {
    onProgress('detect');

    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${API_BASE}/pipeline`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Pipeline failed: ${response.status}`);
    }

    return await response.json();
}

/**
 * Run the prompt-based generation pipeline.
 * 
 * @param {string} promptText - Text to generate hologram from
 * @returns {Promise<Object>} Pipeline result
 */
export async function runPromptPipeline(promptText) {
    const response = await fetch(`${API_BASE}/prompt_pipeline`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: promptText }),
    });

    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Prompt pipeline failed: ${response.status}`);
    }

    return await response.json();
}

/**
 * Detect objects in an image.
 * 
 * @param {File|Blob} imageFile
 * @returns {Promise<Object>} Detection results
 */
export async function detectObjects(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Detection failed: ${response.status}`);
    }

    return await response.json();
}

/**
 * Retrieve a matching 3D model from the catalog.
 * 
 * @param {Object} options
 * @param {File|Blob} [options.imageFile]
 * @param {string} [options.query]
 * @param {number[]} [options.embedding]
 * @returns {Promise<Object>} Retrieval results
 */
export async function retrieveModel({ imageFile, query, embedding } = {}) {
    const formData = new FormData();
    
    if (imageFile) formData.append('file', imageFile);
    if (query) formData.append('query', query);
    if (embedding) formData.append('embedding', JSON.stringify(embedding));

    const response = await fetch(`${API_BASE}/retrieve-model`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Retrieval failed: ${response.status}`);
    }

    return await response.json();
}

/**
 * Generate a 3D model from an image.
 * 
 * @param {File|Blob} imageFile
 * @param {string} [label='object']
 * @returns {Promise<Object>} Generation results with model URL
 */
export async function generateModel(imageFile, label = 'object') {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('label', label);

    const response = await fetch(`${API_BASE}/generate-model`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Generation failed: ${response.status}`);
    }

    return await response.json();
}

/**
 * Get the full URL for a model file served by the backend.
 * @param {string} modelPath - Relative model path from API response
 * @returns {string} Full URL
 */
export function getModelUrl(modelPath) {
    if (!modelPath) return null;
    if (modelPath.startsWith('http')) return modelPath;
    return `${API_BASE}${modelPath}`;
}
