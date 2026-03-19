"""
Configuration and constants for the Hologram Engine backend.
"""
import os
import torch

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CATALOG_DIR = os.path.join(BASE_DIR, "catalog")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CATALOG_DIR, exist_ok=True)

# ─── Device ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] Using device: {DEVICE}")

# ─── YOLOv8 ─────────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8n.pt"  # Nano for speed; use yolov8x.pt for accuracy
YOLO_CONFIDENCE_THRESHOLD = 0.30

# ─── OpenCLIP ────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# ─── Semantic refinement ────────────────────────────────
# Minimum CLIP score required to OVERRIDE/REFINE a YOLO label
SEMANTIC_REFINEMENT_THRESHOLD = 0.28

# ─── Retrieval ───────────────────────────────────────────
RETRIEVAL_SIMILARITY_THRESHOLD = 0.18
RETRIEVAL_TOP_K = 5

# ─── 3D Generation ──────────────────────────────────────
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"
TRIPOSR_CHUNK_SIZE = 8192
TRIPOSR_MC_RESOLUTION = 256

# ─── Objaverse Catalog ──────────────────────────────────
CATALOG_SIZE = 200
CATALOG_EMBEDDINGS_FILE = os.path.join(CATALOG_DIR, "embeddings.npz")
CATALOG_METADATA_FILE = os.path.join(CATALOG_DIR, "metadata.json")

# ─── API ─────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


# ═══════════════════════════════════════════════════════════
# CRITICAL: YOLO label classification
# ═══════════════════════════════════════════════════════════

# YOLO labels that are AMBIGUOUS and need CLIP refinement.
# For example "person" could be Spider-Man, Batman, a soldier, etc.
# Any label NOT in this set is treated as specific/reliable.
YOLO_AMBIGUOUS_LABELS = {
    "person",       # Could be any character, profession, etc.
    "bird",         # Could be eagle, parrot, penguin, etc.
    "cat",          # Could be specific breed or cartoon cat
    "dog",          # Could be specific breed
    "horse",        # Could be zebra, unicorn, etc.
    "bear",         # Could be teddy bear, polar bear, etc.
    "sports ball",  # Could be football, basketball, etc.
}

# Focused refinement categories for each ambiguous YOLO label.
# These are ONLY compared when YOLO says "person", "bird", etc.
REFINEMENT_CATEGORIES = {
    "person": [
        # Superheroes
        "Spider-Man", "Batman", "Superman", "Iron Man", "Captain America",
        "Wonder Woman", "Hulk", "Thor", "Deadpool", "Wolverine",
        "Black Panther", "Aquaman", "Flash", "Green Lantern",
        "Thanos", "Joker", "Harley Quinn",
        # Professions & types
        "a soldier", "a doctor", "a nurse", "a firefighter",
        "a police officer", "an astronaut", "a chef", "a clown",
        "a pirate", "a zombie", "a ninja", "a samurai",
        "a construction worker", "a wizard", "a king", "a queen",
        # Generic fallback (should rank low for characters)
        "a man", "a woman", "a child", "a person standing",
    ],
    "bird": [
        "an eagle", "a parrot", "a penguin", "an owl", "a flamingo",
        "a hawk", "a sparrow", "a crow", "a pigeon", "a peacock",
        "a bird",
    ],
    "cat": [
        "a domestic cat", "a kitten", "a Persian cat", "a Siamese cat",
        "a cartoon cat", "a black cat", "a tabby cat", "a cat",
        "a lion", "a tiger", "a cheetah", "a leopard", "a panther",
    ],
    "dog": [
        "a Labrador", "a German Shepherd", "a Poodle", "a Bulldog",
        "a Golden Retriever", "a Husky", "a Dalmatian", "a Corgi",
        "a Chihuahua", "a Pitbull", "a puppy", "a dog",
        "a wolf", "a fox", "a coyote", "a hyena",
        "a lion", "a tiger", "a bear", "a polar bear" 
    ],
    "horse": [
        "a horse", "a pony", "a zebra", "a donkey", "a unicorn",
        "a race horse", "a wild horse", "a deer", "a moose", "a camel"
    ],
    "bear": [
        "a teddy bear", "a grizzly bear", "a polar bear", "a panda",
        "a koala", "a bear", "a gorilla"
    ],
    "sports ball": [
        "a football", "a soccer ball", "a basketball", "a baseball",
        "a tennis ball", "a volleyball", "a golf ball", "a rugby ball",
        "a bowling ball", "a cricket ball",
    ],
}


# ═══════════════════════════════════════════════════════════
# Full semantic categories (used as fallback when no YOLO)
# ═══════════════════════════════════════════════════════════

SEMANTIC_CATEGORIES = [
    # ── All COCO classes (80 classes that YOLO can detect) ──
    "a person", "a bicycle", "a car", "a motorcycle", "an airplane",
    "a bus", "a train", "a truck", "a boat", "a traffic light",
    "a fire hydrant", "a stop sign", "a parking meter", "a bench",
    "a bird", "a cat", "a dog", "a horse", "a sheep", "a cow",
    "an elephant", "a bear", "a zebra", "a giraffe", "a backpack",
    "an umbrella", "a handbag", "a tie", "a suitcase", "a frisbee",
    "a pair of skis", "a snowboard", "a sports ball", "a kite",
    "a baseball bat", "a baseball glove", "a skateboard", "a surfboard",
    "a tennis racket", "a bottle", "a wine glass", "a cup", "a fork",
    "a knife", "a spoon", "a bowl", "a banana", "an apple",
    "a sandwich", "an orange", "broccoli", "a carrot", "a hot dog",
    "a pizza", "a donut", "a cake", "a chair", "a couch",
    "a potted plant", "a bed", "a dining table", "a toilet",
    "a TV", "a laptop", "a mouse", "a remote control", "a keyboard",
    "a cell phone", "a microwave", "an oven", "a toaster", "a sink",
    "a refrigerator", "a book", "a clock", "a vase", "scissors",
    "a teddy bear", "a hair dryer", "a toothbrush",

    # ── Characters (for refinement) ──
    "Spider-Man", "Batman", "Superman", "Iron Man",

    # ── Additional Animals & Objects ──
    "a lion", "a tiger", "a wolf", "a fox", "a deer", "a monkey", 
    "a shark", "a whale", "a dolphin", "a snake", "a frog",
    "a guitar", "a piano", "a drum", "a lamp", "a mirror",
    "a robot", "a dragon", "a sword", "a shield",
    "a hammer", "a wrench", "a key", "a shoe", "a hat",
    "a flower", "a tree", "a house", "a castle", "a tower",
    "a rocket", "a globe", "a diamond", "a crown", "a trophy",
    "a skull", "a candle", "a pillow", "a basket", "a barrel",
]
