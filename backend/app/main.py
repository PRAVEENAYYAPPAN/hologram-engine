"""
Hologram Engine — FastAPI Backend

Main application with endpoints:
  POST /detect           → object detection + semantic classification
  POST /retrieve-model   → retrieve matching 3D model from catalog
  POST /generate-model   → generate 3D model from image
  GET  /health           → health check
  GET  /models/{filename} → serve generated model files
  POST /pipeline         → full end-to-end pipeline
"""
import io
import os
import time
import traceback

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import API_HOST, API_PORT, CORS_ORIGINS, OUTPUT_DIR

# NOTE: Heavy AI modules (detection, semantic, generation) are imported lazily
# inside their endpoint functions to prevent loading torch/CLIP at startup.
# This keeps startup memory under 512MB on Render's free tier.
# Only retrieval's load_catalog (which reads a small JSON) is imported here.
from .retrieval import load_catalog

from PIL import Image

# ─── App ─────────────────────────────────────────────────
app = FastAPI(
    title="Hologram Engine API",
    description="Real-time object detection → semantic understanding → 3D model retrieval/generation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.on_event("startup")
async def startup():
    """Pre-load models and catalog on startup."""
    print("=" * 60)
    print("🔮 Hologram Engine v2 starting up...")
    print("=" * 60)
    try:
        load_catalog()
    except Exception as e:
        print(f"[Startup] Catalog loading deferred: {e}")


# ─── Health Check ────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "hologram-engine",
        "version": "2.0.0",
        "timestamp": time.time(),
    }


# ─── Detection Endpoint ─────────────────────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detect objects in an uploaded image and classify them semantically.

    IMPORTANT: Uses YOLO-aware classification.
    - Specific YOLO labels (umbrella, bottle) are trusted.
    - Ambiguous YOLO labels (person) are refined by CLIP.
    """
    try:
        from .detection import detect_objects, crop_detection
        from .semantic import classify_semantic, get_image_embedding_from_bytes, get_image_embedding

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        start_time = time.time()

        # Step 1: YOLOv8 object detection
        detections = detect_objects(image_bytes)

        results = []

        if detections:
            for det in detections[:5]:
                # Crop the detected region
                cropped = crop_detection(image_bytes, det["bbox"])

                # CRITICAL: Pass YOLO label + confidence to semantic classifier
                semantic = classify_semantic(
                    cropped,
                    yolo_label=det["label"],
                    yolo_confidence=det["confidence"],
                )

                embedding = get_image_embedding(cropped)

                results.append({
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "semantic_class": semantic["semantic_class"],
                    "semantic_confidence": semantic["semantic_confidence"],
                    "semantic_source": semantic["source"],
                    "top_5_semantic": semantic["top_5"],
                    "embedding": embedding.tolist(),
                })
        else:
            # No YOLO detections — classify the whole image with CLIP only
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            semantic = classify_semantic(image)  # No YOLO context
            image_embedding = get_image_embedding_from_bytes(image_bytes).tolist()

            results.append({
                "label": "unknown",
                "confidence": 0.0,
                "bbox": [],
                "semantic_class": semantic["semantic_class"],
                "semantic_confidence": semantic["semantic_confidence"],
                "semantic_source": semantic["source"],
                "top_5_semantic": semantic["top_5"],
                "embedding": image_embedding,
            })

        elapsed = round(time.time() - start_time, 3)

        return {
            "success": True,
            "detections": results,
            "count": len(results),
            "processing_time_seconds": elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# ─── Model Retrieval Endpoint ────────────────────────────
@app.post("/retrieve-model")
async def retrieve_model_endpoint(
    file: UploadFile = File(None),
    query: str = Form(""),
    embedding: str = Form(""),
):
    """Retrieve the best-matching 3D model from the catalog."""
    try:
        import json
        import numpy as np
        from .semantic import get_image_embedding_from_bytes
        from .retrieval import retrieve_model, download_objaverse_model

        start_time = time.time()
        query_embedding = None
        query_text = query

        if file and file.filename:
            image_bytes = await file.read()
            if image_bytes:
                query_embedding = get_image_embedding_from_bytes(image_bytes)

        if query_embedding is None and embedding:
            query_embedding = np.array(json.loads(embedding), dtype=np.float32)

        if query_embedding is None and query_text:
            from .semantic import get_text_embedding
            query_embedding = get_text_embedding(query_text)

        if query_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="Provide an image file, embedding, or text query"
            )

        match = retrieve_model(query_embedding, query_text)

        if match is None:
            return {
                "success": False,
                "message": "No matching model found above threshold",
                "should_generate": True,
                "processing_time_seconds": round(time.time() - start_time, 3),
            }

        model_path = None
        if not match.get("is_fallback", False):
            model_path = download_objaverse_model(match["uid"])

        elapsed = round(time.time() - start_time, 3)

        return {
            "success": True,
            "match": {
                "name": match["name"],
                "uid": match["uid"],
                "similarity": match["similarity"],
                "tags": match.get("tags", []),
                "categories": match.get("categories", []),
            },
            "model_url": f"/models/{os.path.basename(model_path)}" if model_path else None,
            "should_generate": model_path is None,
            "alternatives": match.get("alternatives", [])[:3],
            "processing_time_seconds": elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


# ─── Model Generation Endpoint ──────────────────────────
@app.post("/generate-model")
async def generate_model_endpoint(
    file: UploadFile = File(...),
    label: str = Form("object"),
):
    """Generate a 3D model from an uploaded image."""
    try:
        from .generation import generate_3d_model

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        start_time = time.time()
        result = generate_3d_model(image_bytes, label)
        elapsed = round(time.time() - start_time, 3)

        if result["success"]:
            return {
                "success": True,
                "model_url": f"/models/{result['model_filename']}",
                "model_filename": result["model_filename"],
                "method": result["method"],
                "processing_time_seconds": elapsed,
            }
        else:
            raise HTTPException(status_code=500, detail="Model generation failed")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ─── Serve Generated Models ─────────────────────────────
@app.get("/models/{filename}")
async def serve_model(filename: str):
    """Serve a generated/downloaded 3D model file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(
        filepath,
        media_type="model/gltf-binary",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600",
        },
    )

class PromptRequest(BaseModel):
    prompt: str

@app.post("/prompt_pipeline")
async def process_prompt_pipeline(req: PromptRequest):
    """
    Lightweight text-prompt → 3D model pipeline.
    
    Uses direct text matching against the fallback catalog instead of
    heavy CLIP embeddings. This keeps memory under 512MB for free tier.
    """
    try:
        prompt = req.prompt.strip().lower()
        if not prompt:
            raise HTTPException(status_code=400, detail="Empty prompt")

        start_time = time.time()

        # ── Direct text matching against our 65-object catalog ──
        catalog_path = os.path.join(
            os.path.dirname(__file__), "..", "catalog", "metadata.json"
        )
        catalog_path = os.path.normpath(catalog_path)

        best_match = None
        best_score = 0.0

        if os.path.exists(catalog_path):
            import json as json_mod
            with open(catalog_path, "r") as f:
                catalog = json_mod.load(f)

            for entry in catalog:
                name = entry.get("name", "").lower()
                tags = [t.lower() for t in entry.get("tags", [])]
                desc = entry.get("description", "").lower()

                score = 0.0
                if prompt == name:
                    score = 1.0
                elif prompt in name or name in prompt:
                    score = 0.8
                elif any(prompt == tag for tag in tags):
                    score = 0.7
                elif any(prompt in tag or tag in prompt for tag in tags):
                    score = 0.5
                elif prompt in desc:
                    score = 0.4

                if score > best_score:
                    best_score = score
                    best_match = entry

        if not best_match or best_score < 0.3:
            raise HTTPException(
                status_code=404,
                detail=f"No 3D model found for '{prompt}'. Try: car, dog, cat, robot, guitar, chair, laptop, sword, dragon, rocket, etc."
            )

        elapsed = round(time.time() - start_time, 3)

        return {
            "success": True,
            "detection": {
                "yolo_label": prompt,
                "confidence": 1.0,
                "semantic_class": best_match["name"].upper(),
                "semantic_confidence": best_score,
                "semantic_source": "text_match"
            },
            "model": {
                "url": None,
                "method": "text_match",
                "match_name": best_match["name"],
                "match_similarity": best_score
            },
            "processing_time_seconds": elapsed
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prompt Pipeline failed: {str(e)}")


# ─── Full Pipeline Endpoint ─────────────────────────────
@app.post("/pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    """
    Run the full pipeline: detect → semantics → retrieve → generate → serve.

    CRITICAL LOGIC:
    1. YOLO detects the object (e.g. "umbrella")
    2. Semantic classifier decides the FINAL label:
       - Specific YOLO labels → trusted as-is (umbrella stays umbrella)
       - Ambiguous labels → refined by CLIP (person → Spider-Man)
    3. The FINAL label drives both retrieval and generation.
    """
    try:
        from .detection import detect_objects, crop_detection
        from .semantic import classify_semantic, get_image_embedding_from_bytes, get_image_embedding
        from .retrieval import retrieve_model, download_objaverse_model
        from .generation import generate_3d_model

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        start_time = time.time()

        # ── Step 1: YOLOv8 Detection ──
        detections = detect_objects(image_bytes)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if detections:
            det = detections[0]
            cropped = crop_detection(image_bytes, det["bbox"])

            # CRITICAL: Pass YOLO label to semantic classifier
            semantic = classify_semantic(
                cropped,
                yolo_label=det["label"],
                yolo_confidence=det["confidence"],
            )

            embedding = get_image_embedding(cropped)
            primary_label = semantic["semantic_class"]
            yolo_label = det["label"]
            confidence = det["confidence"]
        else:
            semantic = classify_semantic(image)  # Pure CLIP
            embedding = get_image_embedding_from_bytes(image_bytes)
            primary_label = semantic["semantic_class"]
            yolo_label = "unknown"
            confidence = 0.0

        print(f"[Pipeline] YOLO: {yolo_label} ({confidence:.2f}) → Semantic: {primary_label} (source: {semantic['source']})")

        # ── Step 2: Retrieval ──
        import numpy as np
        match = retrieve_model(embedding, primary_label)

        model_url = None
        model_method = None

        if match:
            # Try to download real Objaverse model
            if match.get("is_objaverse", False):
                print(f"[Pipeline] 🎯 Found real Objaverse model: '{match['name']}'")
                model_path = download_objaverse_model(match["uid"])
                if model_path:
                    model_url = f"/models/{os.path.basename(model_path)}"
                    model_method = "objaverse"
                    print(f"[Pipeline] ✅ Real model loaded: {model_path}")
            elif not match.get("is_fallback", False):
                model_path = download_objaverse_model(match["uid"])
                if model_path:
                    model_url = f"/models/{os.path.basename(model_path)}"
                    model_method = "retrieval"

        # ── Step 3: Generate if retrieval failed ──
        if model_url is None:
            gen_result = generate_3d_model(image_bytes, primary_label)
            if gen_result["success"]:
                model_url = f"/models/{gen_result['model_filename']}"
                model_method = gen_result["method"]

        elapsed = round(time.time() - start_time, 3)

        return {
            "success": True,
            "detection": {
                "yolo_label": yolo_label,
                "confidence": confidence,
                "semantic_class": primary_label,
                "semantic_confidence": semantic["semantic_confidence"],
                "semantic_source": semantic["source"],
                "top_5_semantic": semantic["top_5"],
            },
            "model": {
                "url": model_url,
                "method": model_method,
                "match_name": match["name"] if match else None,
                "match_similarity": match["similarity"] if match else None,
            },
            "processing_time_seconds": elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


# ─── Run ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)
