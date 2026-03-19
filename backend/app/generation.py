"""
3D Model Generation.

Fallback pipeline: when no suitable model is found in the catalog,
generate a 3D model from a single image using TripoSR (open-source),
or fall back to procedural geometry that MATCHES the object type.
"""
import io
import os
import uuid
import math
import numpy as np
from PIL import Image

from .config import (
    OUTPUT_DIR,
    DEVICE,
    TRIPOSR_MC_RESOLUTION,
)

_triposr_model = None


def get_triposr_model():
    """Lazy-load TripoSR model."""
    global _triposr_model
    if _triposr_model is None:
        print("[Generation] Loading TripoSR model...")
        try:
            from tsr import TSR
            _triposr_model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            _triposr_model.renderer.set_chunk_size(8192)
            _triposr_model.to(DEVICE)
            print("[Generation] TripoSR model loaded.")
        except ImportError:
            print("[Generation] TripoSR not available. Using fallback generation.")
            _triposr_model = "FALLBACK"
        except Exception as e:
            print(f"[Generation] TripoSR loading failed: {e}")
            _triposr_model = "FALLBACK"
    return _triposr_model


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image for better 3D generation."""
    try:
        from rembg import remove
        output = remove(image)
        return output
    except ImportError:
        return image


def generate_3d_model(image_bytes: bytes, label: str = "object") -> dict:
    """
    Generate a 3D model from a single image.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG).
        label: Object label for naming and geometry selection.

    Returns:
        Dict with model_path, model_filename, method, success.
    """
    model_id = str(uuid.uuid4())[:8]
    safe_label = label.lower().replace(" ", "_").replace("-", "_")
    # Remove leading articles
    for prefix in ["a_", "an_", "the_"]:
        if safe_label.startswith(prefix):
            safe_label = safe_label[len(prefix):]
    output_filename = f"{safe_label}_{model_id}.glb"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    model = get_triposr_model()

    if model != "FALLBACK":
        try:
            return _generate_with_triposr(image, output_path)
        except Exception as e:
            print(f"[Generation] TripoSR generation failed: {e}")
            print("[Generation] Falling back to procedural generation.")

    # Fallback: generate object-specific procedural GLB
    return _generate_procedural_glb(image, output_path, label)


def _generate_with_triposr(image: Image.Image, output_path: str) -> dict:
    """Generate 3D model using TripoSR."""
    model = get_triposr_model()
    processed_image = remove_background(image)

    with __import__("torch").no_grad():
        scene_codes = model([processed_image], device=DEVICE)
        meshes = model.extract_mesh(scene_codes, resolution=TRIPOSR_MC_RESOLUTION)

    mesh = meshes[0]
    mesh.export(output_path)

    return {
        "model_path": output_path,
        "model_filename": os.path.basename(output_path),
        "method": "triposr",
        "success": True,
    }


def _generate_procedural_glb(image: Image.Image, output_path: str, label: str) -> dict:
    """
    Generate a procedural GLB file with geometry that MATCHES the object.
    
    Instead of always producing a generic cube, this creates
    object-specific geometry (umbrella → cone+cylinder, bottle → cylinder,
    chair → legs+seat, etc.)
    """
    import struct

    # Get object-specific geometry
    vertices, indices, normals = _get_geometry_for_label(label)

    # Extract dominant color from image
    color = _get_dominant_color(image)

    # Build GLB binary
    glb_bytes = _build_glb(vertices, indices, normals, color)

    with open(output_path, "wb") as f:
        f.write(glb_bytes)

    return {
        "model_path": output_path,
        "model_filename": os.path.basename(output_path),
        "method": "procedural",
        "success": True,
    }


def _get_dominant_color(image: Image.Image) -> list[float]:
    """Extract dominant color from the image using k-means-like approach."""
    img_small = image.resize((50, 50))
    pixels = np.array(img_small).reshape(-1, 3).astype(float)
    
    # Filter out very dark and very light pixels (background)
    mask = (pixels.sum(axis=1) > 60) & (pixels.sum(axis=1) < 700)
    if mask.sum() > 10:
        pixels = pixels[mask]
    
    avg_color = pixels.mean(axis=0) / 255.0
    return avg_color.tolist()


# ═══════════════════════════════════════════════════════════
# OBJECT-SPECIFIC GEOMETRY GENERATORS
# ═══════════════════════════════════════════════════════════

def _get_geometry_for_label(label: str) -> tuple:
    """
    Return appropriate geometry based on the detected object label.
    
    This is a comprehensive mapping that produces recognizable shapes
    for each object type, not generic cubes.
    """
    label_lower = label.lower().replace("a ", "").replace("an ", "").replace("the ", "").strip()

    # ── Direct geometry mapping ──
    geometry_map = {
        # Umbrella: cone canopy + thin pole
        "umbrella": lambda: _make_umbrella(),
        
        # Bottles/containers: cylinder
        "bottle": lambda: _make_cylinder(0.15, 0.6, 16),
        "wine glass": lambda: _make_wine_glass(),
        "cup": lambda: _make_cylinder(0.18, 0.3, 16),
        "mug": lambda: _make_cylinder(0.18, 0.3, 16),
        "glass": lambda: _make_cylinder(0.12, 0.4, 16),
        "vase": lambda: _make_vase(),
        "can": lambda: _make_cylinder(0.12, 0.3, 16),
        "bowl": lambda: _make_hemisphere(0.3, 16, 8),
        "plate": lambda: _make_cylinder(0.35, 0.03, 24),

        # Furniture
        "chair": lambda: _make_chair(),
        "table": lambda: _make_table(),
        "dining table": lambda: _make_table(),
        "desk": lambda: _make_table(),
        "bed": lambda: _make_box(1.2, 0.3, 0.8),
        "couch": lambda: _make_couch(),
        "sofa": lambda: _make_couch(),
        "bench": lambda: _make_box(1.0, 0.3, 0.3),
        "bookshelf": lambda: _make_box(0.8, 1.2, 0.25),
        "cabinet": lambda: _make_box(0.6, 0.8, 0.35),
        "lamp": lambda: _make_lamp(),

        # Electronics
        "laptop": lambda: _make_laptop(),
        "television": lambda: _make_box(1.0, 0.6, 0.05),
        "tv": lambda: _make_box(1.0, 0.6, 0.05),
        "cell phone": lambda: _make_box(0.08, 0.16, 0.01),
        "smartphone": lambda: _make_box(0.08, 0.16, 0.01),
        "phone": lambda: _make_box(0.08, 0.16, 0.01),
        "keyboard": lambda: _make_box(0.45, 0.03, 0.15),
        "mouse": lambda: _make_hemisphere(0.06, 12, 6),
        "computer mouse": lambda: _make_hemisphere(0.06, 12, 6),
        "remote control": lambda: _make_box(0.05, 0.2, 0.02),
        "camera": lambda: _make_box(0.12, 0.08, 0.08),
        "microwave": lambda: _make_box(0.4, 0.25, 0.3),
        "oven": lambda: _make_box(0.5, 0.4, 0.4),
        "toaster": lambda: _make_box(0.2, 0.15, 0.1),
        "refrigerator": lambda: _make_box(0.5, 1.2, 0.5),
        "sink": lambda: _make_box(0.5, 0.2, 0.35),

        # Vehicles
        "car": lambda: _make_car(),
        "truck": lambda: _make_box(0.8, 0.4, 0.3),
        "bus": lambda: _make_box(1.2, 0.5, 0.3),
        "motorcycle": lambda: _make_box(0.6, 0.35, 0.15),
        "motorbike": lambda: _make_box(0.6, 0.35, 0.15),
        "bicycle": lambda: _make_box(0.5, 0.35, 0.1),
        "airplane": lambda: _make_airplane(),
        "boat": lambda: _make_boat(),
        "ship": lambda: _make_boat(),
        "train": lambda: _make_box(1.2, 0.4, 0.3),
        "helicopter": lambda: _make_box(0.5, 0.25, 0.2),
        "rocket": lambda: _make_rocket(),

        # Food (mostly sphere/cylinder based)
        "apple": lambda: _make_sphere(0.12, 16, 16),
        "orange": lambda: _make_sphere(0.12, 16, 16),
        "banana": lambda: _make_banana(),
        "pizza": lambda: _make_cylinder(0.3, 0.03, 24),
        "cake": lambda: _make_cylinder(0.25, 0.15, 24),
        "donut": lambda: _make_torus(0.12, 0.04, 16, 12),
        "hot dog": lambda: _make_cylinder(0.04, 0.25, 12),
        "burger": lambda: _make_cylinder(0.15, 0.1, 16),
        "sandwich": lambda: _make_box(0.15, 0.06, 0.12),
        "broccoli": lambda: _make_hemisphere(0.1, 12, 6),
        "carrot": lambda: _make_cone(0.04, 0.25, 12),

        # Animals — approximate body shapes
        "bird": lambda: _make_sphere(0.12, 12, 12),
        "cat": lambda: _make_animal_body(0.15, 0.12, 0.3),
        "dog": lambda: _make_animal_body(0.18, 0.15, 0.35),
        "horse": lambda: _make_animal_body(0.3, 0.35, 0.6),
        "elephant": lambda: _make_animal_body(0.5, 0.4, 0.6),
        "bear": lambda: _make_animal_body(0.3, 0.3, 0.4),
        "sheep": lambda: _make_animal_body(0.2, 0.2, 0.3),
        "cow": lambda: _make_animal_body(0.3, 0.3, 0.5),
        "zebra": lambda: _make_animal_body(0.25, 0.3, 0.5),
        "giraffe": lambda: _make_box(0.15, 0.8, 0.2),

        # People & Characters — humanoid shapes
        "person": lambda: _make_humanoid(),
        "Spider-Man": lambda: _make_humanoid(),
        "Batman": lambda: _make_humanoid(),
        "Superman": lambda: _make_humanoid(),
        "Iron Man": lambda: _make_humanoid(),
        "Captain America": lambda: _make_humanoid(),
        "Hulk": lambda: _make_humanoid_big(),
        "Thor": lambda: _make_humanoid(),
        "astronaut": lambda: _make_humanoid(),
        "soldier": lambda: _make_humanoid(),
        "robot": lambda: _make_humanoid(),

        # Misc objects
        "book": lambda: _make_box(0.2, 0.25, 0.03),
        "clock": lambda: _make_cylinder(0.2, 0.03, 24),
        "watch": lambda: _make_cylinder(0.05, 0.01, 24),
        "scissors": lambda: _make_box(0.05, 0.2, 0.01),
        "teddy bear": lambda: _make_sphere(0.2, 12, 12),
        "hair dryer": lambda: _make_box(0.1, 0.15, 0.06),
        "toothbrush": lambda: _make_box(0.015, 0.2, 0.015),
        "backpack": lambda: _make_box(0.25, 0.35, 0.12),
        "handbag": lambda: _make_box(0.25, 0.2, 0.08),
        "suitcase": lambda: _make_box(0.35, 0.25, 0.1),
        "tie": lambda: _make_box(0.06, 0.4, 0.01),
        "shoe": lambda: _make_box(0.1, 0.08, 0.28),
        "hat": lambda: _make_cylinder(0.2, 0.1, 16),
        "ball": lambda: _make_sphere(0.15, 16, 16),
        "sports ball": lambda: _make_sphere(0.15, 16, 16),
        "frisbee": lambda: _make_cylinder(0.2, 0.02, 24),
        "skateboard": lambda: _make_box(0.2, 0.03, 0.7),
        "surfboard": lambda: _make_box(0.15, 0.03, 0.8),
        "kite": lambda: _make_box(0.4, 0.01, 0.4),
        "key": lambda: _make_box(0.03, 0.08, 0.005),
        "sword": lambda: _make_box(0.03, 0.6, 0.005),
        "shield": lambda: _make_cylinder(0.25, 0.03, 24),
        "hammer": lambda: _make_box(0.08, 0.3, 0.04),
        "wrench": lambda: _make_box(0.04, 0.2, 0.01),
        "screwdriver": lambda: _make_box(0.02, 0.2, 0.02),
        "tree": lambda: _make_tree(),
        "flower": lambda: _make_flower(),
        "potted plant": lambda: _make_flower(),
        "house": lambda: _make_house(),
        "castle": lambda: _make_box(0.6, 0.7, 0.6),
        "tower": lambda: _make_box(0.15, 0.8, 0.15),
        "fire hydrant": lambda: _make_cylinder(0.08, 0.3, 12),
        "stop sign": lambda: _make_cylinder(0.2, 0.01, 8),
        "traffic light": lambda: _make_box(0.06, 0.25, 0.06),
        "parking meter": lambda: _make_cylinder(0.04, 0.4, 12),
        "toilet": lambda: _make_box(0.25, 0.3, 0.3),
        "globe": lambda: _make_sphere(0.2, 16, 16),
        "diamond": lambda: _make_diamond(),
        "crown": lambda: _make_cylinder(0.12, 0.08, 16),
        "trophy": lambda: _make_wine_glass(),
        "skull": lambda: _make_sphere(0.15, 16, 16),
        "candle": lambda: _make_cylinder(0.03, 0.2, 12),
        "light bulb": lambda: _make_sphere(0.08, 12, 12),
        "dragon": lambda: _make_animal_body(0.3, 0.25, 0.5),
        "dinosaur": lambda: _make_animal_body(0.3, 0.3, 0.5),
    }

    # Try direct match
    if label_lower in geometry_map:
        return geometry_map[label_lower]()

    # Try partial match
    for key, gen_fn in geometry_map.items():
        if key in label_lower or label_lower in key:
            return gen_fn()

    # Ultimate fallback: a box based on general category
    return _make_box(0.4, 0.4, 0.4)


# ════════════════════════════════════════════════════
# COMPOSITE OBJECT GENERATORS
# ════════════════════════════════════════════════════

def _make_umbrella():
    """Umbrella: cone canopy on top + thin cylinder handle."""
    # Canopy (wide cone)
    v1, i1, n1 = _make_cone(0.5, 0.35, 24)
    # Shift canopy up
    for j in range(1, len(v1), 3):
        v1[j] += 0.5
    
    # Handle (thin cylinder)
    v2, i2, n2 = _make_cylinder(0.015, 0.7, 8)
    # Offset indices for second mesh
    offset = len(v1) // 3
    i2 = [idx + offset for idx in i2]
    
    return v1 + v2, i1 + i2, n1 + n2


def _make_wine_glass():
    """Wine glass: bowl on top of a narrow stem on a base."""
    # Bowl (hemisphere on top)
    v1, i1, n1 = _make_hemisphere(0.1, 12, 6)
    for j in range(1, len(v1), 3):
        v1[j] += 0.25

    # Stem (thin cylinder)
    v2, i2, n2 = _make_cylinder(0.015, 0.2, 8)
    offset = len(v1) // 3
    i2 = [idx + offset for idx in i2]
    for j in range(1, len(v2), 3):
        v2[j] += 0.05

    # Base (flat disc)
    v3, i3, n3 = _make_cylinder(0.08, 0.01, 12)
    offset2 = (len(v1) + len(v2)) // 3
    i3 = [idx + offset2 for idx in i3]

    return v1 + v2 + v3, i1 + i2 + i3, n1 + n2 + n3


def _make_vase():
    """Vase: wide bottom, narrow neck."""
    # Body
    v1, i1, n1 = _make_cylinder(0.15, 0.3, 16)
    for j in range(1, len(v1), 3):
        v1[j] += 0.15
    
    # Neck
    v2, i2, n2 = _make_cylinder(0.08, 0.15, 12)
    offset = len(v1) // 3
    i2 = [idx + offset for idx in i2]
    for j in range(1, len(v2), 3):
        v2[j] += 0.4

    return v1 + v2, i1 + i2, n1 + n2


def _make_chair():
    """Chair: seat + backrest + 4 legs."""
    v_all, i_all, n_all = [], [], []
    
    # Seat
    sv, si, sn = _make_box(0.4, 0.04, 0.4)
    for j in range(1, len(sv), 3):
        sv[j] += 0.35
    v_all += sv; i_all += si; n_all += sn
    
    # Backrest
    bv, bi, bn = _make_box(0.4, 0.35, 0.04)
    offset = len(v_all) // 3
    bi = [idx + offset for idx in bi]
    for j in range(1, len(bv), 3):
        bv[j] += 0.55
    for j in range(2, len(bv), 3):
        bv[j] -= 0.18
    v_all += bv; i_all += bi; n_all += bn
    
    # 4 Legs
    for lx, lz in [(-0.15, -0.15), (0.15, -0.15), (-0.15, 0.15), (0.15, 0.15)]:
        lv, li, ln = _make_box(0.03, 0.35, 0.03)
        offset = len(v_all) // 3
        li = [idx + offset for idx in li]
        for j in range(0, len(lv), 3):
            lv[j] += lx
        for j in range(1, len(lv), 3):
            lv[j] += 0.175
        for j in range(2, len(lv), 3):
            lv[j] += lz
        v_all += lv; i_all += li; n_all += ln
    
    return v_all, i_all, n_all


def _make_table():
    """Table: flat top + 4 legs."""
    v_all, i_all, n_all = [], [], []
    
    # Top
    tv, ti, tn = _make_box(0.8, 0.04, 0.5)
    for j in range(1, len(tv), 3):
        tv[j] += 0.55
    v_all += tv; i_all += ti; n_all += tn
    
    # 4 Legs
    for lx, lz in [(-0.35, -0.2), (0.35, -0.2), (-0.35, 0.2), (0.35, 0.2)]:
        lv, li, ln = _make_box(0.03, 0.55, 0.03)
        offset = len(v_all) // 3
        li = [idx + offset for idx in li]
        for j in range(0, len(lv), 3):
            lv[j] += lx
        for j in range(1, len(lv), 3):
            lv[j] += 0.275
        for j in range(2, len(lv), 3):
            lv[j] += lz
        v_all += lv; i_all += li; n_all += ln
    
    return v_all, i_all, n_all


def _make_couch():
    """Couch: wide seat + back + armrests."""
    v_all, i_all, n_all = [], [], []
    
    # Seat
    sv, si, sn = _make_box(1.0, 0.15, 0.4)
    for j in range(1, len(sv), 3):
        sv[j] += 0.2
    v_all += sv; i_all += si; n_all += sn
    
    # Backrest
    bv, bi, bn = _make_box(1.0, 0.3, 0.08)
    offset = len(v_all) // 3
    bi = [idx + offset for idx in bi]
    for j in range(1, len(bv), 3):
        bv[j] += 0.42
    for j in range(2, len(bv), 3):
        bv[j] -= 0.16
    v_all += bv; i_all += bi; n_all += bn
    
    # Armrests
    for side in [-1, 1]:
        av, ai, an = _make_box(0.05, 0.2, 0.4)
        offset = len(v_all) // 3
        ai = [idx + offset for idx in ai]
        for j in range(0, len(av), 3):
            av[j] += side * 0.5
        for j in range(1, len(av), 3):
            av[j] += 0.35
        v_all += av; i_all += ai; n_all += an
    
    return v_all, i_all, n_all


def _make_lamp():
    """Lamp: shade (cone) + pole + base."""
    v_all, i_all, n_all = [], [], []
    
    # Shade
    sv, si, sn = _make_cone(0.2, 0.15, 16)
    for j in range(1, len(sv), 3):
        sv[j] += 0.55
    v_all += sv; i_all += si; n_all += sn
    
    # Pole
    pv, pi, pn = _make_cylinder(0.02, 0.5, 8)
    offset = len(v_all) // 3
    pi = [idx + offset for idx in pi]
    for j in range(1, len(pv), 3):
        pv[j] += 0.1
    v_all += pv; i_all += pi; n_all += pn
    
    # Base
    bv, bi, bn = _make_cylinder(0.1, 0.02, 16)
    offset = len(v_all) // 3
    bi = [idx + offset for idx in bi]
    v_all += bv; i_all += bi; n_all += bn
    
    return v_all, i_all, n_all


def _make_laptop():
    """Laptop: two flat boxes hinged."""
    v_all, i_all, n_all = [], [], []
    
    # Base/keyboard
    bv, bi, bn = _make_box(0.35, 0.02, 0.25)
    for j in range(1, len(bv), 3):
        bv[j] += 0.01
    v_all += bv; i_all += bi; n_all += bn
    
    # Screen (tilted)
    sv, si, sn = _make_box(0.33, 0.25, 0.01)
    offset = len(v_all) // 3
    si = [idx + offset for idx in si]
    for j in range(1, len(sv), 3):
        sv[j] += 0.14
    for j in range(2, len(sv), 3):
        sv[j] -= 0.12
    v_all += sv; i_all += si; n_all += sn
    
    return v_all, i_all, n_all


def _make_car():
    """Car: body + cabin."""
    v_all, i_all, n_all = [], [], []
    
    # Body
    bv, bi, bn = _make_box(0.7, 0.15, 0.3)
    for j in range(1, len(bv), 3):
        bv[j] += 0.1
    v_all += bv; i_all += bi; n_all += bn
    
    # Cabin
    cv, ci, cn = _make_box(0.35, 0.15, 0.28)
    offset = len(v_all) // 3
    ci = [idx + offset for idx in ci]
    for j in range(1, len(cv), 3):
        cv[j] += 0.25
    v_all += cv; i_all += ci; n_all += cn
    
    # Wheels (4 small cylinders)
    for wx, wz in [(-0.25, -0.18), (0.25, -0.18), (-0.25, 0.18), (0.25, 0.18)]:
        wv, wi, wn = _make_sphere(0.06, 8, 8)
        offset = len(v_all) // 3
        wi = [idx + offset for idx in wi]
        for j in range(0, len(wv), 3):
            wv[j] += wx
        for j in range(1, len(wv), 3):
            wv[j] += 0.06
        for j in range(2, len(wv), 3):
            wv[j] += wz
        v_all += wv; i_all += wi; n_all += wn
    
    return v_all, i_all, n_all


def _make_airplane():
    """Airplane: fuselage + wings + tail."""
    v_all, i_all, n_all = [], [], []
    
    # Fuselage
    fv, fi, fn = _make_cylinder(0.06, 0.8, 12)
    # Rotate horizontal
    for j in range(0, len(fv), 3):
        fv[j], fv[j+1] = fv[j+1], fv[j]
    for j in range(0, len(fn), 3):
        fn[j], fn[j+1] = fn[j+1], fn[j]
    for j in range(1, len(fv), 3):
        fv[j] += 0.2
    v_all += fv; i_all += fi; n_all += fn
    
    # Wings
    wv, wi, wn = _make_box(0.02, 0.8, 0.15)
    offset = len(v_all) // 3
    wi = [idx + offset for idx in wi]
    for j in range(1, len(wv), 3):
        wv[j] += 0.2
    v_all += wv; i_all += wi; n_all += wn
    
    return v_all, i_all, n_all


def _make_boat():
    """Boat: hull shape."""
    v_all, i_all, n_all = [], [], []
    
    # Hull (box - bottom)
    hv, hi, hn = _make_box(0.6, 0.12, 0.2)
    for j in range(1, len(hv), 3):
        hv[j] += 0.06
    v_all += hv; i_all += hi; n_all += hn
    
    # Cabin
    cv, ci, cn = _make_box(0.15, 0.12, 0.12)
    offset = len(v_all) // 3
    ci = [idx + offset for idx in ci]
    for j in range(1, len(cv), 3):
        cv[j] += 0.18
    v_all += cv; i_all += ci; n_all += cn
    
    return v_all, i_all, n_all


def _make_rocket():
    """Rocket: cylinder body + cone nose."""
    v_all, i_all, n_all = [], [], []
    
    # Body
    bv, bi, bn = _make_cylinder(0.08, 0.5, 12)
    for j in range(1, len(bv), 3):
        bv[j] += 0.25
    v_all += bv; i_all += bi; n_all += bn
    
    # Nose cone
    cv, ci, cn = _make_cone(0.08, 0.15, 12)
    offset = len(v_all) // 3
    ci = [idx + offset for idx in ci]
    for j in range(1, len(cv), 3):
        cv[j] += 0.55
    v_all += cv; i_all += ci; n_all += cn
    
    return v_all, i_all, n_all


def _make_humanoid():
    """Humanoid figure: head + torso + arms + legs."""
    v_all, i_all, n_all = [], [], []
    
    # Head
    hv, hi, hn = _make_sphere(0.1, 10, 10)
    for j in range(1, len(hv), 3):
        hv[j] += 0.85
    v_all += hv; i_all += hi; n_all += hn
    
    # Torso
    tv, ti, tn = _make_box(0.25, 0.35, 0.12)
    offset = len(v_all) // 3
    ti = [idx + offset for idx in ti]
    for j in range(1, len(tv), 3):
        tv[j] += 0.55
    v_all += tv; i_all += ti; n_all += tn
    
    # Arms
    for side in [-1, 1]:
        av, ai, an = _make_box(0.06, 0.3, 0.06)
        offset = len(v_all) // 3
        ai = [idx + offset for idx in ai]
        for j in range(0, len(av), 3):
            av[j] += side * 0.19
        for j in range(1, len(av), 3):
            av[j] += 0.52
        v_all += av; i_all += ai; n_all += an
    
    # Legs
    for side in [-1, 1]:
        lv, li, ln = _make_box(0.08, 0.35, 0.08)
        offset = len(v_all) // 3
        li = [idx + offset for idx in li]
        for j in range(0, len(lv), 3):
            lv[j] += side * 0.08
        for j in range(1, len(lv), 3):
            lv[j] += 0.175
        v_all += lv; i_all += li; n_all += ln
    
    return v_all, i_all, n_all


def _make_humanoid_big():
    """Larger humanoid (Hulk, etc.)."""
    v, i, n = _make_humanoid()
    # Scale up by 1.4x
    v = [x * 1.4 for x in v]
    return v, i, n


def _make_animal_body(width, height, length):
    """Generic animal body: torso + legs + head."""
    v_all, i_all, n_all = [], [], []
    
    # Body
    bv, bi, bn = _make_box(length, height * 0.6, width)
    for j in range(1, len(bv), 3):
        bv[j] += height * 0.6
    v_all += bv; i_all += bi; n_all += bn
    
    # Head
    hv, hi, hn = _make_sphere(height * 0.35, 8, 8)
    offset = len(v_all) // 3
    hi = [idx + offset for idx in hi]
    for j in range(0, len(hv), 3):
        hv[j] += length * 0.4
    for j in range(1, len(hv), 3):
        hv[j] += height * 0.75
    v_all += hv; i_all += hi; n_all += hn
    
    # 4 Legs
    for lx, lz in [(length*0.3, width*0.3), (length*0.3, -width*0.3),
                    (-length*0.3, width*0.3), (-length*0.3, -width*0.3)]:
        lv, li, ln = _make_box(0.03, height * 0.5, 0.03)
        offset = len(v_all) // 3
        li = [idx + offset for idx in li]
        for j in range(0, len(lv), 3):
            lv[j] += lx
        for j in range(1, len(lv), 3):
            lv[j] += height * 0.25
        for j in range(2, len(lv), 3):
            lv[j] += lz
        v_all += lv; i_all += li; n_all += ln
    
    return v_all, i_all, n_all


def _make_tree():
    """Tree: trunk + foliage sphere."""
    v_all, i_all, n_all = [], [], []
    
    # Trunk
    tv, ti, tn = _make_cylinder(0.04, 0.4, 8)
    for j in range(1, len(tv), 3):
        tv[j] += 0.2
    v_all += tv; i_all += ti; n_all += tn
    
    # Foliage
    fv, fi, fn = _make_sphere(0.25, 12, 12)
    offset = len(v_all) // 3
    fi = [idx + offset for idx in fi]
    for j in range(1, len(fv), 3):
        fv[j] += 0.6
    v_all += fv; i_all += fi; n_all += fn
    
    return v_all, i_all, n_all


def _make_flower():
    """Flower: stem + bloom."""
    v_all, i_all, n_all = [], [], []
    
    # Stem
    sv, si, sn = _make_cylinder(0.015, 0.3, 6)
    for j in range(1, len(sv), 3):
        sv[j] += 0.15
    v_all += sv; i_all += si; n_all += sn
    
    # Bloom
    bv, bi, bn = _make_sphere(0.08, 10, 10)
    offset = len(v_all) // 3
    bi = [idx + offset for idx in bi]
    for j in range(1, len(bv), 3):
        bv[j] += 0.4
    v_all += bv; i_all += bi; n_all += bn
    
    # Pot
    pv, pi, pn = _make_cylinder(0.08, 0.08, 12)
    offset = len(v_all) // 3
    pi = [idx + offset for idx in pi]
    for j in range(1, len(pv), 3):
        pv[j] += 0.04
    v_all += pv; i_all += pi; n_all += pn
    
    return v_all, i_all, n_all


def _make_house():
    """House: box body + triangle roof."""
    v_all, i_all, n_all = [], [], []
    
    # Body
    bv, bi, bn = _make_box(0.5, 0.32, 0.4)
    for j in range(1, len(bv), 3):
        bv[j] += 0.16
    v_all += bv; i_all += bi; n_all += bn
    
    # Roof (wider, flatter box tilted — approximation)
    rv, ri, rn = _make_cone(0.4, 0.2, 4)
    offset = len(v_all) // 3
    ri = [idx + offset for idx in ri]
    for j in range(1, len(rv), 3):
        rv[j] += 0.38
    v_all += rv; i_all += ri; n_all += rn
    
    return v_all, i_all, n_all


def _make_diamond():
    """Diamond: two cones facing each other."""
    v_all, i_all, n_all = [], [], []
    
    # Top cone
    tv, ti, tn = _make_cone(0.15, 0.15, 8)
    for j in range(1, len(tv), 3):
        tv[j] += 0.25
    v_all += tv; i_all += ti; n_all += tn
    
    # Bottom inverted cone
    bv, bi, bn = _make_cone(0.15, 0.25, 8)
    offset = len(v_all) // 3
    bi = [idx + offset for idx in bi]
    # Invert Y for bottom
    for j in range(1, len(bv), 3):
        bv[j] = 0.25 - bv[j]
    v_all += bv; i_all += bi; n_all += bn
    
    return v_all, i_all, n_all


def _make_banana():
    """Banana: slightly curved cylinder."""
    # Approximate with a tilted cylinder
    v, i, n = _make_cylinder(0.03, 0.25, 10)
    # Slight curve by bending x based on y
    for j in range(0, len(v), 3):
        y = v[j + 1]
        v[j] += 0.05 * math.sin(y * 6.28 / 0.25)
    for j in range(1, len(v), 3):
        v[j] += 0.125
    return v, i, n


# ════════════════════════════════════════════════════
# PRIMITIVE GEOMETRY BUILDERS
# ════════════════════════════════════════════════════

def _make_box(sx: float, sy: float, sz: float):
    """Create box geometry centered at origin."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    vertices = [
        -hx, -hy,  hz,   hx, -hy,  hz,   hx,  hy,  hz,  -hx,  hy,  hz,
        -hx, -hy, -hz,  -hx,  hy, -hz,   hx,  hy, -hz,   hx, -hy, -hz,
        -hx,  hy, -hz,  -hx,  hy,  hz,   hx,  hy,  hz,   hx,  hy, -hz,
        -hx, -hy, -hz,   hx, -hy, -hz,   hx, -hy,  hz,  -hx, -hy,  hz,
         hx, -hy, -hz,   hx,  hy, -hz,   hx,  hy,  hz,   hx, -hy,  hz,
        -hx, -hy, -hz,  -hx, -hy,  hz,  -hx,  hy,  hz,  -hx,  hy, -hz,
    ]
    normals = [
        0,0,1,  0,0,1,  0,0,1,  0,0,1,
        0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
        0,1,0,  0,1,0,  0,1,0,  0,1,0,
        0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0,
        1,0,0,  1,0,0,  1,0,0,  1,0,0,
        -1,0,0, -1,0,0, -1,0,0, -1,0,0,
    ]
    indices = [
        0,1,2,  0,2,3,
        4,5,6,  4,6,7,
        8,9,10, 8,10,11,
        12,13,14, 12,14,15,
        16,17,18, 16,18,19,
        20,21,22, 20,22,23,
    ]
    return vertices, indices, normals


def _make_sphere(radius: float, lat_segments: int, lon_segments: int):
    """Create sphere geometry."""
    vertices = []
    normals = []
    indices = []

    for lat in range(lat_segments + 1):
        theta = lat * math.pi / lat_segments
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for lon in range(lon_segments + 1):
            phi = lon * 2 * math.pi / lon_segments
            x = math.cos(phi) * sin_theta
            y = cos_theta
            z = math.sin(phi) * sin_theta
            vertices.extend([x * radius, y * radius, z * radius])
            normals.extend([x, y, z])

    for lat in range(lat_segments):
        for lon in range(lon_segments):
            first = lat * (lon_segments + 1) + lon
            second = first + lon_segments + 1
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return vertices, indices, normals


def _make_cylinder(radius: float, height: float, segments: int):
    """Create cylinder geometry."""
    vertices = []
    normals = []
    indices = []

    # Side vertices
    for i in range(segments + 1):
        angle = i * 2 * math.pi / segments
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        nx, nz = math.cos(angle), math.sin(angle)

        # Bottom vertex
        vertices.extend([x, 0, z])
        normals.extend([nx, 0, nz])
        # Top vertex
        vertices.extend([x, height, z])
        normals.extend([nx, 0, nz])

    # Side indices
    for i in range(segments):
        b = i * 2
        indices.extend([b, b + 1, b + 2])
        indices.extend([b + 1, b + 3, b + 2])

    # Top cap center
    top_center = len(vertices) // 3
    vertices.extend([0, height, 0])
    normals.extend([0, 1, 0])

    for i in range(segments):
        angle = i * 2 * math.pi / segments
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        idx = len(vertices) // 3
        vertices.extend([x, height, z])
        normals.extend([0, 1, 0])

        if i > 0:
            indices.extend([top_center, idx - 1, idx])

    indices.extend([top_center, len(vertices) // 3 - 1, top_center + 1])

    # Bottom cap center
    bot_center = len(vertices) // 3
    vertices.extend([0, 0, 0])
    normals.extend([0, -1, 0])

    for i in range(segments):
        angle = i * 2 * math.pi / segments
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        idx = len(vertices) // 3
        vertices.extend([x, 0, z])
        normals.extend([0, -1, 0])

        if i > 0:
            indices.extend([bot_center, idx, idx - 1])

    indices.extend([bot_center, bot_center + 1, len(vertices) // 3 - 1])

    return vertices, indices, normals


def _make_cone(radius: float, height: float, segments: int):
    """Create cone geometry."""
    vertices = []
    normals = []
    indices = []

    # Apex
    vertices.extend([0, height, 0])
    normals.extend([0, 1, 0])

    # Base ring
    for i in range(segments + 1):
        angle = i * 2 * math.pi / segments
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        # Normal pointing outward and upward
        ny = radius / math.sqrt(radius * radius + height * height)
        nr = height / math.sqrt(radius * radius + height * height)
        nx = math.cos(angle) * nr
        nz = math.sin(angle) * nr
        vertices.extend([x, 0, z])
        normals.extend([nx, ny, nz])

    # Side faces
    for i in range(segments):
        indices.extend([0, i + 1, i + 2])

    # Base cap
    base_center = len(vertices) // 3
    vertices.extend([0, 0, 0])
    normals.extend([0, -1, 0])

    for i in range(segments + 1):
        angle = i * 2 * math.pi / segments
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        idx = len(vertices) // 3
        vertices.extend([x, 0, z])
        normals.extend([0, -1, 0])
        if i > 0:
            indices.extend([base_center, idx, idx - 1])

    return vertices, indices, normals


def _make_hemisphere(radius: float, segments: int, rings: int):
    """Create hemisphere (half sphere) geometry."""
    vertices = []
    normals = []
    indices = []

    for lat in range(rings + 1):
        theta = lat * (math.pi / 2) / rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for lon in range(segments + 1):
            phi = lon * 2 * math.pi / segments
            x = math.cos(phi) * sin_theta
            y = cos_theta
            z = math.sin(phi) * sin_theta
            vertices.extend([x * radius, y * radius, z * radius])
            normals.extend([x, y, z])

    for lat in range(rings):
        for lon in range(segments):
            first = lat * (segments + 1) + lon
            second = first + segments + 1
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return vertices, indices, normals


def _make_torus(major_radius: float, minor_radius: float, major_segments: int, minor_segments: int):
    """Create torus (donut) geometry."""
    vertices = []
    normals = []
    indices = []

    for i in range(major_segments + 1):
        theta = i * 2 * math.pi / major_segments
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        for j in range(minor_segments + 1):
            phi = j * 2 * math.pi / minor_segments
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)

            x = (major_radius + minor_radius * cos_phi) * cos_theta
            y = minor_radius * sin_phi
            z = (major_radius + minor_radius * cos_phi) * sin_theta

            nx = cos_phi * cos_theta
            ny = sin_phi
            nz = cos_phi * sin_theta

            vertices.extend([x, y, z])
            normals.extend([nx, ny, nz])

    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * (minor_segments + 1) + j
            b = a + minor_segments + 1
            indices.extend([a, b, a + 1])
            indices.extend([b, b + 1, a + 1])

    return vertices, indices, normals


# ════════════════════════════════════════════════════
# GLB BUILDER
# ════════════════════════════════════════════════════

def _build_glb(vertices: list, indices: list, normals: list, color: list[float]) -> bytes:
    """Build a complete GLB (GLTF Binary) file."""
    import struct
    import json as json_mod

    vertex_data = struct.pack(f"<{len(vertices)}f", *vertices)
    normal_data = struct.pack(f"<{len(normals)}f", *normals)
    index_data = struct.pack(f"<{len(indices)}H", *indices)

    while len(index_data) % 4 != 0:
        index_data += b"\x00"

    buffer_data = vertex_data + normal_data + index_data

    num_verts = len(vertices) // 3
    min_vals = [float("inf")] * 3
    max_vals = [float("-inf")] * 3
    for i in range(num_verts):
        for j in range(3):
            v = vertices[i * 3 + j]
            min_vals[j] = min(min_vals[j], v)
            max_vals[j] = max(max_vals[j], v)

    vertex_byte_length = len(vertex_data)
    normal_byte_length = len(normal_data)
    index_byte_length = len(indices) * 2

    gltf = {
        "asset": {"version": "2.0", "generator": "HologramEngine"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "HologramMesh"}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1},
                "indices": 2,
                "material": 0,
            }]
        }],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorFactor": [color[0], color[1], color[2], 1.0],
                "metallicFactor": 0.3,
                "roughnessFactor": 0.7,
            },
            "emissiveFactor": [
                min(color[0] * 0.3, 1.0),
                min(color[1] * 0.3, 1.0),
                min(color[2] * 0.3, 1.0),
            ],
            "name": "HologramMaterial",
        }],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126,
                "count": num_verts, "type": "VEC3",
                "min": min_vals, "max": max_vals,
            },
            {
                "bufferView": 1, "componentType": 5126,
                "count": num_verts, "type": "VEC3",
            },
            {
                "bufferView": 2, "componentType": 5123,
                "count": len(indices), "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": vertex_byte_length, "target": 34962},
            {"buffer": 0, "byteOffset": vertex_byte_length, "byteLength": normal_byte_length, "target": 34962},
            {"buffer": 0, "byteOffset": vertex_byte_length + normal_byte_length, "byteLength": index_byte_length, "target": 34963},
        ],
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    gltf_json = json_mod.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(gltf_json) % 4 != 0:
        gltf_json += b" "

    total_length = 12 + 8 + len(gltf_json) + 8 + len(buffer_data)

    glb = bytearray()
    glb += struct.pack("<I", 0x46546C67)
    glb += struct.pack("<I", 2)
    glb += struct.pack("<I", total_length)
    glb += struct.pack("<I", len(gltf_json))
    glb += struct.pack("<I", 0x4E4F534A)
    glb += gltf_json
    glb += struct.pack("<I", len(buffer_data))
    glb += struct.pack("<I", 0x004E4942)
    glb += buffer_data

    return bytes(glb)
