"""
3D Model Retrieval — Real Objaverse models + CLIP-based search.

This module ACTUALLY downloads real 3D models from Objaverse,
not just text-matching against a fallback list.

Architecture:
  1. On startup: load Objaverse annotations in background thread
  2. On query: search annotations by name/tags for matching models
  3. Download the actual .glb file from Objaverse CDN
  4. Cache it locally and serve through the API

This gives REAL detailed 3D models (proper body shapes, armor,
furniture details, etc.) instead of primitive cubes/spheres.
"""
import os
import json
import shutil
import threading
import time
import numpy as np
from pathlib import Path

from .config import (
    CATALOG_DIR,
    CATALOG_EMBEDDINGS_FILE,
    CATALOG_METADATA_FILE,
    OUTPUT_DIR,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K,
)
# NOTE: semantic import is lazy to avoid loading torch/open_clip at startup

# ═══════════════════════════════════════════════════════════
# OBJAVERSE STATE
# ═══════════════════════════════════════════════════════════

_objaverse_annotations = None
_objaverse_loading = False
_objaverse_ready = False

_catalog_embeddings = None
_catalog_metadata = None

# Cache of downloaded model paths: uid -> local_path
_model_cache = {}


# ═══════════════════════════════════════════════════════════
# OBJAVERSE BACKGROUND LOADER
# ═══════════════════════════════════════════════════════════

def _load_objaverse_background():
    """Load Objaverse annotations in a background thread."""
    global _objaverse_annotations, _objaverse_loading, _objaverse_ready

    _objaverse_loading = True
    try:
        import objaverse
        print("[Objaverse] Loading annotations (first time downloads ~100MB, then cached)...")
        start = time.time()
        _objaverse_annotations = objaverse.load_annotations()
        elapsed = time.time() - start
        print(f"[Objaverse] ✅ Loaded {len(_objaverse_annotations)} model annotations in {elapsed:.1f}s")
        _objaverse_ready = True
    except Exception as e:
        print(f"[Objaverse] ⚠️ Loading failed: {e}")
        print("[Objaverse] Will use fallback catalog only.")
    _objaverse_loading = False


def start_objaverse_loading():
    """Start loading Objaverse annotations in background."""
    thread = threading.Thread(target=_load_objaverse_background, daemon=True)
    thread.start()


# ═══════════════════════════════════════════════════════════
# OBJAVERSE SEARCH — Find real models by text
# ═══════════════════════════════════════════════════════════

def search_objaverse(query: str, max_results: int = 10) -> list[dict]:
    """
    Search Objaverse annotations for models matching a text query.
    
    Uses keyword matching on model names and tags.
    Returns list of dicts with uid, name, tags.
    """
    if not _objaverse_ready or _objaverse_annotations is None:
        return []

    query_lower = query.lower().strip()
    # Remove articles
    for prefix in ["a ", "an ", "the "]:
        if query_lower.startswith(prefix):
            query_lower = query_lower[len(prefix):]

    query_words = query_lower.split()

    results = []
    scores = []

    for uid, anno in _objaverse_annotations.items():
        name = anno.get("name", "").strip()
        if not name or len(name) < 2:
            continue

        name_lower = name.lower()
        tags = [t.get("name", "").lower() for t in anno.get("tags", [])]
        categories = [c.get("name", "").lower() for c in anno.get("categories", [])]
        all_text = name_lower + " " + " ".join(tags) + " " + " ".join(categories)

        # Score: how many query words match
        score = 0
        for word in query_words:
            if word in name_lower:
                score += 3  # Name match is strongest
            elif any(word in t for t in tags):
                score += 2  # Tag match
            elif any(word in c for c in categories):
                score += 1  # Category match

        # Bonus for exact name match
        if query_lower == name_lower:
            score += 10
        elif query_lower in name_lower:
            score += 5

        if score > 0:
            # Prefer models with more metadata (likely higher quality)
            quality_bonus = min(len(tags), 5) * 0.1
            # Prefer shorter names (often more specific/clean models)
            name_penalty = max(0, len(name) - 30) * 0.01

            final_score = score + quality_bonus - name_penalty
            results.append({
                "uid": uid,
                "name": name,
                "tags": [t.get("name", "") for t in anno.get("tags", [])][:5],
                "categories": [c.get("name", "") for c in anno.get("categories", [])][:3],
                "score": final_score,
            })
            scores.append(final_score)

    # Sort by score descending
    if results:
        sorted_pairs = sorted(zip(scores, results), key=lambda x: -x[0])
        results = [r for _, r in sorted_pairs]

    return results[:max_results]


# ═══════════════════════════════════════════════════════════
# OBJAVERSE MODEL DOWNLOAD
# ═══════════════════════════════════════════════════════════

def download_objaverse_model(uid: str) -> str | None:
    """
    Download an actual 3D model file from Objaverse.
    
    Returns the local file path to the .glb file, or None on failure.
    Downloaded models are cached in the output directory.
    """
    # Check cache first
    if uid in _model_cache and os.path.exists(_model_cache[uid]):
        return _model_cache[uid]

    # Check if already downloaded to output dir
    cached_path = os.path.join(OUTPUT_DIR, f"objaverse_{uid}.glb")
    if os.path.exists(cached_path):
        _model_cache[uid] = cached_path
        return cached_path

    try:
        import objaverse
        print(f"[Objaverse] Downloading model: {uid}...")
        start = time.time()

        objects = objaverse.load_objects(uids=[uid])

        if uid in objects:
            src_path = objects[uid]
            # Copy to our output directory
            shutil.copy2(src_path, cached_path)
            _model_cache[uid] = cached_path
            elapsed = time.time() - start
            print(f"[Objaverse] ✅ Downloaded model in {elapsed:.1f}s → {cached_path}")
            return cached_path
        else:
            print(f"[Objaverse] ⚠️ Model {uid} not found in download results")

    except Exception as e:
        print(f"[Objaverse] ❌ Download failed for {uid}: {e}")

    return None


# ═══════════════════════════════════════════════════════════
# FALLBACK CATALOG
# ═══════════════════════════════════════════════════════════

def _build_fallback_catalog():
    """Build text-embedding based fallback catalog for when Objaverse is unavailable."""
    global _catalog_embeddings, _catalog_metadata

    fallback_objects = [
        {"name": "person", "tags": ["human", "people", "standing"]},
        {"name": "bicycle", "tags": ["bike", "cycling"]},
        {"name": "car", "tags": ["vehicle", "automobile"]},
        {"name": "motorcycle", "tags": ["motorbike", "vehicle"]},
        {"name": "airplane", "tags": ["aircraft", "jet"]},
        {"name": "bus", "tags": ["vehicle", "transport"]},
        {"name": "train", "tags": ["vehicle", "railway"]},
        {"name": "truck", "tags": ["vehicle", "cargo"]},
        {"name": "boat", "tags": ["vessel", "ship"]},
        {"name": "traffic light", "tags": ["signal", "road"]},
        {"name": "fire hydrant", "tags": ["hydrant", "street"]},
        {"name": "stop sign", "tags": ["sign", "road"]},
        {"name": "bench", "tags": ["seat", "park"]},
        {"name": "bird", "tags": ["animal", "flying"]},
        {"name": "cat", "tags": ["animal", "pet"]},
        {"name": "dog", "tags": ["animal", "pet"]},
        {"name": "horse", "tags": ["animal", "equine"]},
        {"name": "sheep", "tags": ["animal", "farm"]},
        {"name": "cow", "tags": ["animal", "farm"]},
        {"name": "elephant", "tags": ["animal", "large"]},
        {"name": "bear", "tags": ["animal", "wild"]},
        {"name": "zebra", "tags": ["animal", "striped"]},
        {"name": "giraffe", "tags": ["animal", "tall"]},
        {"name": "backpack", "tags": ["bag", "travel"]},
        {"name": "umbrella", "tags": ["rain", "canopy"]},
        {"name": "handbag", "tags": ["bag", "purse"]},
        {"name": "tie", "tags": ["necktie", "clothing"]},
        {"name": "suitcase", "tags": ["luggage", "travel"]},
        {"name": "bottle", "tags": ["container", "drink"]},
        {"name": "cup", "tags": ["mug", "drink"]},
        {"name": "bowl", "tags": ["dish", "food"]},
        {"name": "chair", "tags": ["furniture", "seat"]},
        {"name": "couch", "tags": ["sofa", "furniture"]},
        {"name": "bed", "tags": ["furniture", "sleeping"]},
        {"name": "dining table", "tags": ["table", "furniture"]},
        {"name": "toilet", "tags": ["bathroom", "plumbing"]},
        {"name": "television", "tags": ["TV", "screen"]},
        {"name": "laptop", "tags": ["computer", "portable"]},
        {"name": "keyboard", "tags": ["typing", "keys"]},
        {"name": "cell phone", "tags": ["smartphone", "mobile"]},
        {"name": "microwave", "tags": ["kitchen", "appliance"]},
        {"name": "oven", "tags": ["kitchen", "cooking"]},
        {"name": "refrigerator", "tags": ["fridge", "kitchen"]},
        {"name": "book", "tags": ["reading", "pages"]},
        {"name": "clock", "tags": ["time", "mechanical"]},
        {"name": "vase", "tags": ["container", "flowers"]},
        {"name": "scissors", "tags": ["cutting", "tool"]},
        {"name": "teddy bear", "tags": ["toy", "stuffed"]},
        {"name": "Spider-Man", "tags": ["superhero", "marvel"]},
        {"name": "Batman", "tags": ["superhero", "dc"]},
        {"name": "Superman", "tags": ["superhero", "dc"]},
        {"name": "Iron Man", "tags": ["superhero", "marvel"]},
        {"name": "guitar", "tags": ["instrument", "music"]},
        {"name": "piano", "tags": ["instrument", "music"]},
        {"name": "lamp", "tags": ["light", "illumination"]},
        {"name": "robot", "tags": ["machine", "android"]},
        {"name": "dragon", "tags": ["fantasy", "creature"]},
        {"name": "sword", "tags": ["weapon", "blade"]},
        {"name": "tree", "tags": ["nature", "plant"]},
        {"name": "house", "tags": ["building", "home"]},
        {"name": "rocket", "tags": ["space", "vehicle"]},
        {"name": "globe", "tags": ["earth", "world"]},
        {"name": "crown", "tags": ["royalty", "gold"]},
        {"name": "skull", "tags": ["bone", "skeleton"]},
        {"name": "diamond", "tags": ["gem", "crystal"]},
    ]

    embeddings = []
    metadata = []
    for obj in fallback_objects:
        desc = obj["name"] + " " + " ".join(obj["tags"])
        emb = get_text_embedding(desc)
        embeddings.append(emb)
        metadata.append({
            "uid": f"fallback_{obj['name'].lower().replace(' ', '_')}",
            "name": obj["name"],
            "tags": obj["tags"],
            "categories": [],
            "description": desc,
            "is_fallback": True,
        })

    _catalog_embeddings = np.array(embeddings)
    _catalog_metadata = metadata

    np.savez_compressed(CATALOG_EMBEDDINGS_FILE, embeddings=_catalog_embeddings)
    with open(CATALOG_METADATA_FILE, "w") as f:
        json.dump(_catalog_metadata, f, indent=2)

    print(f"[Retrieval] Fallback catalog ready: {len(metadata)} objects.")


# ═══════════════════════════════════════════════════════════
# CATALOG LOADING
# ═══════════════════════════════════════════════════════════

def load_catalog():
    """Load or build the catalog. Objaverse loading is disabled on free tier to save RAM."""
    global _catalog_embeddings, _catalog_metadata

    if _catalog_embeddings is not None:
        return

    # Check for cached catalog on disk
    if os.path.exists(CATALOG_EMBEDDINGS_FILE) and os.path.exists(CATALOG_METADATA_FILE):
        print("[Retrieval] Loading cached catalog from disk...")
        data = np.load(CATALOG_EMBEDDINGS_FILE)
        _catalog_embeddings = data["embeddings"]
        with open(CATALOG_METADATA_FILE, "r") as f:
            _catalog_metadata = json.load(f)
        print(f"[Retrieval] Catalog loaded: {len(_catalog_metadata)} objects.")
    else:
        _build_fallback_catalog()

    # NOTE: Objaverse loading disabled to stay within 512MB RAM on free tier.
    # The fallback CLIP-embedding catalog provides instant retrieval for common objects.
    # To enable Objaverse, uncomment the next line on a higher-RAM instance:
    # start_objaverse_loading()


# ═══════════════════════════════════════════════════════════
# MAIN RETRIEVAL FUNCTION
# ═══════════════════════════════════════════════════════════

def retrieve_model(query_embedding: np.ndarray, query_text: str = "") -> dict:
    """
    Find the best-matching 3D model for a query.
    
    Strategy:
      1. If Objaverse is ready → search for REAL models by name
      2. If not → use text-embedding based fallback catalog
    
    Returns dict with match info, or None if nothing found.
    """
    load_catalog()

    # ── Strategy 1: Search Objaverse for real models ──
    if _objaverse_ready and query_text:
        objaverse_results = search_objaverse(query_text)

        if objaverse_results:
            best = objaverse_results[0]
            print(f"[Retrieval] 🎯 Objaverse match: '{best['name']}' (score: {best['score']:.1f})")

            return {
                "uid": best["uid"],
                "name": best["name"],
                "tags": best["tags"],
                "categories": best["categories"],
                "similarity": round(best["score"] / 10.0, 4),  # Normalize
                "is_fallback": False,
                "is_objaverse": True,
                "alternatives": [
                    {
                        "uid": r["uid"],
                        "name": r["name"],
                        "similarity": round(r["score"] / 10.0, 4),
                    }
                    for r in objaverse_results[1:4]
                ],
            }

    # ── Strategy 2: Fallback catalog (text embeddings) ──
    if _catalog_embeddings is None or len(_catalog_embeddings) == 0:
        return None

    query_emb = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

    similarities = _catalog_embeddings @ query_emb

    if query_text:
        text_emb = get_text_embedding(query_text)
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        text_sim = _catalog_embeddings @ text_emb
        similarities = 0.3 * similarities + 0.7 * text_sim

    top_indices = np.argsort(similarities)[::-1][:RETRIEVAL_TOP_K]
    best_idx = top_indices[0]
    best_score = float(similarities[best_idx])

    if best_score < RETRIEVAL_SIMILARITY_THRESHOLD:
        return None

    match = _catalog_metadata[best_idx].copy()
    match["similarity"] = round(best_score, 4)
    match["is_objaverse"] = False
    match["alternatives"] = []

    for idx in top_indices[1:]:
        alt = _catalog_metadata[idx].copy()
        alt["similarity"] = round(float(similarities[idx]), 4)
        match["alternatives"].append(alt)

    return match
