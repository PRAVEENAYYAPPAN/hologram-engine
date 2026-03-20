/**
 * Hologram Renderer Module — v3
 * 
 * World-class holographic engine with texture-aware shaders,
 * dynamic scaling/rotation correction, depth complexity, and bloom.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

let scene, camera, renderer, controls, composer;
let holoGroup = null;
let particleSystem = null;
let platformGroup = null;
let currentModel = null;
let wireframeOverlay = null;
let animationId = null;
let autoRotate = true;
let wireframeMode = false;
let canvasEl = null;
let clock = null;

let platformRings = [];
let platformGlowDisc = null;
let platformDataRing = null;

const sharedUniforms = {
    time: { value: 0.0 },
};

// ════════════════════════════════════════════════════════
// INITIALIZATION & SCENE SETUP
// ════════════════════════════════════════════════════════

export function init(canvas) {
    canvasEl = canvas;
    clock = new THREE.Clock();

    scene = new THREE.Scene();
    scene.background = null; 
    scene.fog = new THREE.FogExp2(0x020208, 0.05);

    camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
    camera.position.set(0, 1.5, 3.5);

    renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance',
    });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0; 
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Post-processing for intense but controlled glow
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(canvas.clientWidth, canvas.clientHeight),
        0.25,     // slightly bumped strength to keep the cyber feel
        0.5,     // radius
        1.2      // Threshold > 1.0 ensures standard texture colors never "blow out"
    );
    composer.addPass(bloomPass);
    composer.addPass(new OutputPass());

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.0;
    controls.minDistance = 0.1; // Allow close zooming
    controls.maxDistance = 20;  // Allow far zooming
    controls.target.set(0, 0.7, 0);
    controls.maxPolarAngle = Math.PI * 0.85;
    controls.update();

    setupLights();
    setupPlatform();
    setupParticles();

    holoGroup = new THREE.Group();
    scene.add(holoGroup);

    const ro = new ResizeObserver(() => handleResize());
    ro.observe(canvas.parentElement);

    animate();
}

function setupLights() {
    scene.add(new THREE.AmbientLight(0x111122, 0.8));
    
    const mainLight = new THREE.PointLight(0x00f0ff, 2, 10);
    mainLight.position.set(0, 4, 0);
    scene.add(mainLight);

    const fillLight = new THREE.PointLight(0x2288ff, 1.5, 8);
    fillLight.position.set(-3, 2, 2);
    scene.add(fillLight);

    const backLight = new THREE.PointLight(0xaa22ff, 1.0, 8);
    backLight.position.set(2, 1, -3);
    scene.add(backLight);
}

function setupPlatform() {
    platformGroup = new THREE.Group();
    scene.add(platformGroup);

    // Deep tech floor gradient
    const discGeo = new THREE.CircleGeometry(1.5, 64);
    const discMat = new THREE.ShaderMaterial({
        uniforms: {
            time: sharedUniforms.time,
            color: { value: new THREE.Color(0x00f0ff) },
        },
        vertexShader: `
            varying vec2 vUv;
            void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
        `,
        fragmentShader: `
            uniform float time; uniform vec3 color; varying vec2 vUv;
            void main() {
                vec2 center = vUv - 0.5; float dist = length(center);
                float glow = smoothstep(0.6, 0.0, dist) * 0.15;
                float ring1 = smoothstep(0.015, 0.0, abs(dist - mod(time * 0.2, 0.5)));
                float ring2 = smoothstep(0.01, 0.0, abs(dist - mod(time * 0.2 + 0.25, 0.5)));
                float gridX = smoothstep(0.01, 0.0, abs(mod(center.x * 30.0, 1.0) - 0.5) - 0.48);
                float gridY = smoothstep(0.01, 0.0, abs(mod(center.y * 30.0, 1.0) - 0.5) - 0.48);
                float grid = (gridX + gridY) * 0.04 * smoothstep(0.6, 0.1, dist);
                gl_FragColor = vec4(color, glow + ring1 * 0.2 + ring2 * 0.1 + grid);
            }
        `,
        transparent: true, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    platformGlowDisc = new THREE.Mesh(discGeo, discMat);
    platformGlowDisc.rotation.x = -Math.PI / 2;
    platformGlowDisc.position.y = -0.01;
    platformGroup.add(platformGlowDisc);

    // Multi-layered rings
    const ringRadii = [0.4, 0.7, 1.0, 1.3];
    const ringWidths = [0.008, 0.012, 0.005, 0.01];
    for (let i = 0; i < ringRadii.length; i++) {
        const geo = new THREE.RingGeometry(ringRadii[i] - ringWidths[i], ringRadii[i] + ringWidths[i], 64);
        const mat = new THREE.MeshBasicMaterial({
            color: i % 2 === 0 ? 0x00f0ff : 0x0066ff,
            transparent: true, opacity: 0.5 - i * 0.08,
            side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
        });
        const ring = new THREE.Mesh(geo, mat);
        ring.rotation.x = -Math.PI / 2;
        platformRings.push(ring);
        platformGroup.add(ring);
    }
}

function setupParticles() {
    const count = 1000;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);

    const c1 = new THREE.Color(0x00f0ff);
    const c2 = new THREE.Color(0x4466ff);

    for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        const angle = Math.random() * Math.PI * 2;
        const radius = 0.2 + Math.random() * 3.0;
        positions[i3] = Math.cos(angle) * radius;
        positions[i3 + 1] = Math.random() * 4.0 - 0.5;
        positions[i3 + 2] = Math.sin(angle) * radius;

        const color = Math.random() > 0.5 ? c1 : c2;
        colors[i3] = color.r; colors[i3 + 1] = color.g; colors[i3 + 2] = color.b;
        sizes[i] = 0.005 + Math.random() * 0.015;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const mat = new THREE.PointsMaterial({
        size: 0.02, vertexColors: true, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    });
    particleSystem = new THREE.Points(geo, mat);
    scene.add(particleSystem);
}


// ════════════════════════════════════════════════════════
// TEXTURE-AWARE HOLOGRAPHIC SHADERS
// ════════════════════════════════════════════════════════

const holoVertexShader = `
    varying vec3 vNormal;
    varying vec3 vWorldPosition;
    varying vec3 vViewDir;
    varying vec2 vUv;

    #ifdef USE_COLOR
    varying vec3 vColor;
    #endif

    void main() {
        #ifdef USE_COLOR
        vColor = color;
        #endif
        
        vNormal = normalize(normalMatrix * normal);
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPos.xyz;
        vViewDir = normalize(cameraPosition - worldPos.xyz);
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const holoFragmentShader = `
    uniform float time;
    uniform vec3 baseColor;
    uniform vec3 rimColor;
    uniform float opacity;
    uniform sampler2D map;
    uniform bool useMap;

    varying vec3 vNormal;
    varying vec3 vWorldPosition;
    varying vec3 vViewDir;
    varying vec2 vUv;

    #ifdef USE_COLOR
    varying vec3 vColor;
    #endif

    void main() {
        vec3 sourceColor = baseColor;
        
        #ifdef USE_COLOR
        sourceColor *= vColor;
        #endif

        if (useMap) {
            vec4 tex = texture2D(map, vUv);
            sourceColor = tex.rgb;
        }

        // Extremely subtle cyan tint (5%) to preserve pure photographic color
        vec3 holoTint = vec3(0.0, 0.7, 1.0);
        vec3 finalBase = mix(sourceColor, holoTint, 0.05);

        // Prevent absolute black patches
        float luminance = dot(sourceColor, vec3(0.299, 0.587, 0.114));
        if (luminance < 0.05) finalBase += holoTint * 0.1;

        vec3 color = finalBase;

        // Subtle Fresnel Rim (only 15% strength)
        float fresnel = pow(1.0 - abs(dot(vViewDir, vNormal)), 3.0);
        color += rimColor * fresnel * 0.15;

        // Scanlines
        float scanline = sin(vWorldPosition.y * 200.0 - time * 2.0) * 0.5 + 0.5;
        scanline = smoothstep(0.4, 0.6, scanline);
        // Instead of adding pure white/cyan, we just gently multiply or add slightly
        color += holoTint * scanline * 0.1;

        // Vertical scanning band
        float band = smoothstep(0.05, 0.0, abs(vWorldPosition.y - (mod(time * 0.8, 2.0) - 1.0)));
        color += holoTint * band * 0.2;

        float finalAlpha = opacity;
        finalAlpha = clamp(finalAlpha, 0.25, 1.0);

        gl_FragColor = vec4(color, finalAlpha);
    }
`;

function createHolographicMaterial(originalMap = null, originalColor = null, hasVertexColors = false) {
    return new THREE.ShaderMaterial({
        uniforms: {
            time: sharedUniforms.time,
            baseColor: { value: originalColor || new THREE.Color(0x33bbee) },
            rimColor: { value: new THREE.Color(0x00ffff) },
            opacity: { value: 0.8 }, 
            map: { value: originalMap },
            useMap: { value: !!originalMap }
        },
        vertexShader: holoVertexShader,
        fragmentShader: holoFragmentShader,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false, 
        blending: THREE.NormalBlending,
        vertexColors: hasVertexColors
    });
}

const wireFragmentShader = `
    uniform float time;
    uniform vec3 wireColor;
    varying vec3 vWorldPosition;
    void main() {
        float band = smoothstep(0.4, 0.0, abs(vWorldPosition.y - (mod(time * 0.5, 3.0) - 1.0)));
        gl_FragColor = vec4(wireColor, 0.05 + band * 0.2); 
    }
`;

function createWireframeMaterial() {
    return new THREE.ShaderMaterial({
        uniforms: {
            time: sharedUniforms.time,
            wireColor: { value: new THREE.Color(0x00f0ff) },
        },
        vertexShader: `
            varying vec3 vWorldPosition;
            void main() {
                vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: wireFragmentShader,
        transparent: true, wireframe: true, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending,
    });
}

// ════════════════════════════════════════════════════════
// MODEL LOADING
// ════════════════════════════════════════════════════════

export function loadModel(url) {
    return new Promise((resolve, reject) => {
        clearModel();

        const loader = new GLTFLoader();
        loader.load(url, (gltf) => {
            const model = gltf.scene;

            // ── FIX 1: UP-RIGHT ALIGNMENT ── 
            // Many models are exported Z-up instead of Y-up or lay flat.
            // We compute the bounding box before rotation to check proportions.
            const initialBox = new THREE.Box3().setFromObject(model);
            const initialSize = initialBox.getSize(new THREE.Vector3());
            
            // If the model is significantly wider and deeper than it is tall, 
            // it's likely lying down. Rotate it upright.
            if (initialSize.y < initialSize.z && initialSize.y < initialSize.x) {
                model.rotation.x = -Math.PI / 2;
                model.updateMatrixWorld(true);
            }

            // Center and Scale
            const box = new THREE.Box3().setFromObject(model);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 1.8 / maxDim;

            model.scale.set(scale, scale, scale);
            model.position.sub(center.multiplyScalar(scale));
            model.position.y += size.y * scale / 2;

            // ── FIX 2: TEXTURE-AWARE SHADERS & WIREFRAMES ──
            model.traverse((child) => {
                if (child.isMesh) {
                    // Skip invisible collision/hitbox meshes completely
                    if (!child.visible || (child.material && child.material.opacity === 0)) {
                        child.visible = false;
                        return;
                    }

                    child.userData.originalMaterial = child.material;
                    
                    let map = null;
                    let originalColor = null;
                    
                    if (child.material) {
                        if (child.material.map) map = child.material.map;
                        if (child.material.color) originalColor = child.material.color.clone();
                    }
                    
                    const hasVColor = child.geometry && child.geometry.hasAttribute('color');
                    child.material = createHolographicMaterial(map, originalColor, hasVColor);

                    // Create perfectly aligned, clean wireframe without diagonal clutter
                    // EdgesGeometry only draws feature edges, much cleaner for holograms
                    const edges = new THREE.EdgesGeometry(child.geometry, 20); 
                    const lineMat = createWireframeMaterial();
                    const wireLine = new THREE.LineSegments(edges, lineMat);
                    
                    // Add directly to the mesh so it inherits ALL local transforms instantly
                    child.add(wireLine);
                    child.userData.wireframeLine = wireLine;
                }
            });

            currentModel = model;
            holoGroup.add(model);

            const modelHeight = size.y * scale;
            controls.target.set(0, modelHeight / 2, 0);
            camera.position.set(0, modelHeight * 0.6, 3);
            controls.update();

            resolve();
        }, undefined, reject);
    });
}

export function clearModel() {
    if (currentModel) {
        holoGroup.remove(currentModel);
        currentModel.traverse(child => {
            if (child.isMesh || child.isLineSegments) {
                child.geometry?.dispose();
                if (child.material) {
                    Array.isArray(child.material) ? child.material.forEach(m => m.dispose()) : child.material.dispose();
                }
            }
        });
    }
    currentModel = null;
    wireframeOverlay = null; // Removed global variable concept
}

// ════════════════════════════════════════════════════════
// PROCEDURAL HOLOGRAM GENERATION (No GLB needed)
// ════════════════════════════════════════════════════════

const SHAPE_MAP = {
    // Vehicles
    car: 'box', truck: 'box', bus: 'box', train: 'box',
    // Animals
    dog: 'sphere', cat: 'sphere', bird: 'cone', fish: 'torus',
    horse: 'sphere', elephant: 'sphere', bear: 'sphere',
    // Tech
    laptop: 'box', phone: 'box', tv: 'box', keyboard: 'box',
    // Weapons / tools
    sword: 'cylinder', knife: 'cylinder', axe: 'cylinder',
    // Nature
    tree: 'cone', flower: 'torus', mushroom: 'cone',
    // Furniture
    chair: 'box', table: 'box', bed: 'box', couch: 'box',
    // Round objects
    ball: 'sphere', globe: 'sphere', planet: 'sphere', apple: 'sphere',
    // Fantasy
    dragon: 'icosahedron', robot: 'dodecahedron', rocket: 'cone',
    // Music
    guitar: 'cylinder', drum: 'cylinder',
    // Books / flat
    book: 'box',
};

export function generateProcedural(label) {
    clearModel();

    const group = new THREE.Group();
    const name = label.toLowerCase().trim();
    const shapeType = SHAPE_MAP[name] || 'icosahedron';

    // ── Primary shape ──
    let geometry;
    switch (shapeType) {
        case 'box':
            geometry = new THREE.BoxGeometry(1.2, 0.8, 1.0, 4, 4, 4);
            break;
        case 'sphere':
            geometry = new THREE.SphereGeometry(0.7, 32, 32);
            break;
        case 'cone':
            geometry = new THREE.ConeGeometry(0.6, 1.4, 32);
            break;
        case 'cylinder':
            geometry = new THREE.CylinderGeometry(0.15, 0.15, 1.6, 32);
            break;
        case 'torus':
            geometry = new THREE.TorusGeometry(0.5, 0.2, 16, 48);
            break;
        case 'dodecahedron':
            geometry = new THREE.DodecahedronGeometry(0.7, 0);
            break;
        case 'icosahedron':
        default:
            geometry = new THREE.IcosahedronGeometry(0.7, 1);
            break;
    }

    // Main holographic mesh
    const mainMat = createHolographicMaterial(null, new THREE.Color(0x00ddff), false);
    const mesh = new THREE.Mesh(geometry, mainMat);
    mesh.position.y = 1.0;
    group.add(mesh);

    // Wireframe overlay
    const edges = new THREE.EdgesGeometry(geometry, 15);
    const wireMat = createWireframeMaterial();
    const wireframe = new THREE.LineSegments(edges, wireMat);
    mesh.add(wireframe);

    // ── Orbiting rings ──
    for (let i = 0; i < 3; i++) {
        const ringGeo = new THREE.TorusGeometry(0.9 + i * 0.2, 0.008, 8, 64);
        const ringMat = new THREE.MeshBasicMaterial({
            color: i === 0 ? 0x00f0ff : i === 1 ? 0x0088ff : 0x00ffaa,
            transparent: true, opacity: 0.4 - i * 0.08,
            side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false,
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.y = 1.0;
        ring.rotation.x = Math.PI / 2 + (i - 1) * 0.4;
        ring.rotation.z = i * 0.6;
        ring.userData._orbitIndex = i;
        group.add(ring);
    }

    // ── Floating data particles around shape ──
    const pCount = 200;
    const pPositions = new Float32Array(pCount * 3);
    const pColors = new Float32Array(pCount * 3);
    const c1 = new THREE.Color(0x00f0ff);
    const c2 = new THREE.Color(0x00ffaa);
    for (let i = 0; i < pCount; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = 0.8 + Math.random() * 0.6;
        pPositions[i * 3] = Math.cos(angle) * r;
        pPositions[i * 3 + 1] = 0.3 + Math.random() * 1.5;
        pPositions[i * 3 + 2] = Math.sin(angle) * r;
        const c = Math.random() > 0.5 ? c1 : c2;
        pColors[i * 3] = c.r; pColors[i * 3 + 1] = c.g; pColors[i * 3 + 2] = c.b;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
    pGeo.setAttribute('color', new THREE.BufferAttribute(pColors, 3));
    const pMat = new THREE.PointsMaterial({
        size: 0.03, vertexColors: true, transparent: true, opacity: 0.6,
        blending: THREE.AdditiveBlending, depthWrite: false,
    });
    group.add(new THREE.Points(pGeo, pMat));

    // Store animation reference
    group.userData._proceduralMesh = mesh;
    group.userData._isProceduralHologram = true;

    currentModel = group;
    holoGroup.add(group);

    controls.target.set(0, 1.0, 0);
    camera.position.set(0, 1.2, 3);
    controls.update();
}

// ════════════════════════════════════════════════════════
// RENDER LOOP
// ════════════════════════════════════════════════════════

function animate() {
    animationId = requestAnimationFrame(animate);
    const delta = clock.getDelta();
    const elapsed = clock.getElapsedTime();

    sharedUniforms.time.value = elapsed;

    if (currentModel) {
        currentModel.traverse(child => {
            if ((child.isMesh || child.isLineSegments) && child.material && child.material.uniforms && child.material.uniforms.time) {
                child.material.uniforms.time.value = elapsed;
            }
        });

        // Animate procedural elements if this is a procedural hologram
        if (currentModel.userData._isProceduralHologram) {
            currentModel.children.forEach(child => {
                // Orbiting rings
                if (child.userData._orbitIndex !== undefined) {
                    child.rotation.x += delta * (0.2 + child.userData._orbitIndex * 0.1);
                    child.rotation.y += delta * 0.15;
                }
            });
            // Float the main mesh slightly
            if (currentModel.userData._proceduralMesh) {
                currentModel.userData._proceduralMesh.position.y = 1.0 + Math.sin(elapsed * 2.0) * 0.05;
                currentModel.userData._proceduralMesh.rotation.y += delta * 0.5;
                currentModel.userData._proceduralMesh.rotation.z += delta * 0.2;
            }
        }
    }

    if (particleSystem) {
        particleSystem.rotation.y += delta * 0.05;
        const pos = particleSystem.geometry.attributes.position.array;
        for (let i = 0; i < pos.length; i += 3) {
            pos[i + 1] += Math.sin(elapsed * 0.5 + i) * 0.001; 
        }
        particleSystem.geometry.attributes.position.needsUpdate = true;
    }

    platformRings.forEach((ring, i) => {
        ring.rotation.z = elapsed * (0.1 + i * 0.05) * (i % 2 === 0 ? 1 : -1);
    });

    controls.update();
    composer.render();
}

function handleResize() {
    if (!canvasEl || !canvasEl.parentElement) return;
    const { clientWidth: width, clientHeight: height } = canvasEl.parentElement;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    composer.setSize(width, height);
}

export function resetCamera() {
    camera.position.set(0, 1.5, 3.5);
    controls.target.set(0, 0.7, 0);
    controls.update();
}

export function toggleWireframe() {
    wireframeMode = !wireframeMode;
    if (currentModel) {
        currentModel.traverse(child => {
            if (child.isMesh && child.userData.wireframeLine && child.material) {
                // Hide the actual solid material instead of the object, 
                // so the child line segment keeps rendering accurately
                child.material.visible = !wireframeMode;
            }
        });
    }
    return wireframeMode;
}

export function toggleAutoRotate() {
    autoRotate = !autoRotate;
    controls.autoRotate = autoRotate;
    return autoRotate;
}
export function fixOrientation() {
    if (currentModel) {
        currentModel.rotation.x -= Math.PI / 2;
        currentModel.updateMatrixWorld();
    }
}

export function dispose() {
    if (animationId) cancelAnimationFrame(animationId);
    clearModel();
    renderer?.dispose();
    controls?.dispose();
    composer?.dispose();
}
