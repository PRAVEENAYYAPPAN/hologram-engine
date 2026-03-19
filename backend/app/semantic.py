"""
Semantic Understanding using OpenCLIP.

Two-stage classification:
  Stage 1: YOLO gives a coarse label (e.g. "person", "umbrella").
  Stage 2: OpenCLIP refines ONLY when the YOLO label is ambiguous.

This avoids the bug where CLIP overrides a correct YOLO label with
a random category (e.g. umbrella → "Wolverine").
"""
import io
import numpy as np
import torch
from PIL import Image

import open_clip
from .config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DEVICE,
    SEMANTIC_CATEGORIES,
    YOLO_AMBIGUOUS_LABELS,
    SEMANTIC_REFINEMENT_THRESHOLD,
)

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_text_embeddings = None


def get_clip_model():
    """Lazy-load OpenCLIP model, preprocessor, and tokenizer."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        print(f"[Semantic] Loading OpenCLIP: {CLIP_MODEL_NAME} / {CLIP_PRETRAINED}")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        _clip_model = _clip_model.to(DEVICE).eval()
        print("[Semantic] OpenCLIP model loaded.")
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _get_text_embeddings() -> torch.Tensor:
    """Pre-compute and cache text embeddings for all semantic categories."""
    global _text_embeddings
    if _text_embeddings is None:
        model, _, tokenizer = get_clip_model()
        prompts = [f"a photo of {cat}" for cat in SEMANTIC_CATEGORIES]
        tokens = tokenizer(prompts).to(DEVICE)
        with torch.no_grad():
            _text_embeddings = model.encode_text(tokens)
            _text_embeddings = _text_embeddings / _text_embeddings.norm(dim=-1, keepdim=True)
    return _text_embeddings


def get_image_embedding(image: Image.Image) -> np.ndarray:
    """
    Compute the CLIP embedding for an image.

    Args:
        image: PIL Image (RGB).

    Returns:
        Normalized embedding as numpy array (shape: [dim]).
    """
    model, preprocess, _ = get_clip_model()
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


def get_image_embedding_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Convenience wrapper: bytes → embedding."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return get_image_embedding(image)

def get_text_embedding_single(prompt: str) -> np.ndarray:
    """Compute the CLIP embedding for a single text prompt."""
    model, _, tokenizer = get_clip_model()
    tokens = tokenizer([prompt]).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


def _raw_clip_classify(image: Image.Image) -> dict:
    """
    Raw CLIP classification against all semantic categories.
    Returns the full results without any YOLO context.
    """
    model, preprocess, _ = get_clip_model()
    text_embs = _get_text_embeddings()

    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    # Cosine similarity
    similarities = (img_emb @ text_embs.T).squeeze(0).cpu().numpy()

    top_indices = np.argsort(similarities)[::-1][:10]
    top_results = [
        {
            "category": SEMANTIC_CATEGORIES[i],
            "score": round(float(similarities[i]), 4),
        }
        for i in top_indices
    ]

    best_idx = top_indices[0]
    return {
        "best_class": SEMANTIC_CATEGORIES[best_idx],
        "best_score": round(float(similarities[best_idx]), 4),
        "top_results": top_results,
    }


def classify_semantic(image: Image.Image, yolo_label: str = "", yolo_confidence: float = 0.0) -> dict:
    """
    Intelligent two-stage semantic classification.

    Logic:
      1. If YOLO gave a SPECIFIC label (e.g. "umbrella", "bottle", "chair")
         with decent confidence -> TRUST YOLO. Use CLIP only to verify.
      2. If YOLO gave an AMBIGUOUS label (e.g. "person") -> use CLIP to
         refine (e.g. "person" → "Spider-Man" or "astronaut").
      3. If no YOLO label -> rely purely on CLIP.

    This prevents the bug where CLIP overrides a correct YOLO label
    (umbrella → "Wolverine") while still allowing CLIP to refine
    ambiguous labels (person → "Spider-Man").

    Args:
        image: PIL Image (RGB).
        yolo_label: YOLO detection label (e.g. "umbrella", "person").
        yolo_confidence: YOLO confidence score.

    Returns:
        Dict with:
            - semantic_class: best matching category
            - semantic_confidence: similarity score
            - top_5: list of top match dicts
            - source: "yolo" or "clip" (which system made the decision)
    """
    clip_result = _raw_clip_classify(image)
    
    # Case 1: YOLO gave a specific, non-ambiguous label with good confidence
    if yolo_label and yolo_label not in YOLO_AMBIGUOUS_LABELS and yolo_confidence >= 0.25:
        # YOLO is reliable for this object type. Use its label directly.
        # But check if CLIP has a VERY HIGH confidence alternative
        clip_top = clip_result["best_class"]
        clip_score = clip_result["best_score"]

        # Only let CLIP override if it's VERY confident (>0.30) and
        # the CLIP result is a more specific version of the YOLO label
        if clip_score > SEMANTIC_REFINEMENT_THRESHOLD:
            # Check if CLIP is refining the same general category
            yolo_lower = yolo_label.lower()
            clip_lower = clip_top.lower().replace("a ", "").replace("an ", "")
            if yolo_lower in clip_lower or clip_lower in yolo_lower:
                # CLIP is a refinement of YOLO (e.g. "car" → "sports car")
                return {
                    "semantic_class": clip_top,
                    "semantic_confidence": clip_score,
                    "top_5": clip_result["top_results"][:5],
                    "source": "clip_refined",
                }

        # Otherwise trust YOLO — map it to a clean display name
        display_label = _yolo_to_display_name(yolo_label)
        return {
            "semantic_class": display_label,
            "semantic_confidence": round(yolo_confidence, 4),
            "top_5": clip_result["top_results"][:5],
            "source": "yolo",
        }

    # Case 2: YOLO gave an ambiguous label (person, animal, etc.)
    # → Use CLIP to refine
    if yolo_label and yolo_label in YOLO_AMBIGUOUS_LABELS:
        clip_top = clip_result["best_class"]
        clip_score = clip_result["best_score"]

        # Build a refined category list specific to this YOLO class
        refined = _refine_for_yolo_class(image, yolo_label)
        if refined:
            return {
                "semantic_class": refined["best_class"],
                "semantic_confidence": refined["best_score"],
                "top_5": refined["top_results"][:5],
                "source": "clip_refined",
            }

        # Fall back to general CLIP if refinement didn't produce good results
        if clip_score > SEMANTIC_REFINEMENT_THRESHOLD:
            return {
                "semantic_class": clip_top,
                "semantic_confidence": clip_score,
                "top_5": clip_result["top_results"][:5],
                "source": "clip",
            }

        # Last resort: use YOLO label
        display_label = _yolo_to_display_name(yolo_label)
        return {
            "semantic_class": display_label,
            "semantic_confidence": round(yolo_confidence, 4),
            "top_5": clip_result["top_results"][:5],
            "source": "yolo_fallback",
        }

    # Case 3: No YOLO label — rely on CLIP entirely
    return {
        "semantic_class": clip_result["best_class"],
        "semantic_confidence": clip_result["best_score"],
        "top_5": clip_result["top_results"][:5],
        "source": "clip",
    }


def _refine_for_yolo_class(image: Image.Image, yolo_label: str) -> dict | None:
    """
    For ambiguous YOLO labels, run CLIP against a focused subset
    of categories relevant to that class.

    E.g., for "person" → compare against superhero/character categories.
    """
    from .config import REFINEMENT_CATEGORIES

    subcats = REFINEMENT_CATEGORIES.get(yolo_label, [])
    if not subcats:
        return None

    model, preprocess, tokenizer = get_clip_model()
    prompts = [f"a photo of {cat}" for cat in subcats]
    tokens = tokenizer(prompts).to(DEVICE)

    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        text_embs = model.encode_text(tokens)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    similarities = (img_emb @ text_embs.T).squeeze(0).cpu().numpy()
    top_indices = np.argsort(similarities)[::-1][:5]

    best_idx = top_indices[0]
    best_score = float(similarities[best_idx])

    # Only return refined result if score is meaningful
    if best_score < 0.22:
        return None

    top_results = [
        {
            "category": subcats[i],
            "score": round(float(similarities[i]), 4),
        }
        for i in top_indices
    ]

    return {
        "best_class": subcats[best_idx],
        "best_score": round(best_score, 4),
        "top_results": top_results,
    }


def _yolo_to_display_name(yolo_label: str) -> str:
    """Convert YOLO COCO label to a clean display name."""
    # Map YOLO names to nicer display names
    name_map = {
        "tv": "television",
        "cell phone": "smartphone",
        "potted plant": "potted plant",
        "dining table": "table",
        "fire hydrant": "fire hydrant",
        "stop sign": "stop sign",
        "parking meter": "parking meter",
        "sports ball": "ball",
        "baseball bat": "baseball bat",
        "baseball glove": "baseball glove",
        "tennis racket": "tennis racket",
        "wine glass": "wine glass",
        "hot dog": "hot dog",
        "teddy bear": "teddy bear",
        "hair drier": "hair dryer",
    }
    return name_map.get(yolo_label, yolo_label)


def get_text_embedding(text: str) -> np.ndarray:
    """
    Compute the CLIP embedding for a text query.

    Args:
        text: Text string to embed.

    Returns:
        Normalized embedding as numpy array (shape: [dim]).
    """
    model, _, tokenizer = get_clip_model()
    tokens = tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()
