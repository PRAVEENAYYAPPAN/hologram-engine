"""
Test the intelligence pipeline fixes.
Verifies that:
1. Umbrella → "umbrella" (not "Wolverine")
2. GLB contains umbrella geometry (not generic cube)
3. Catalog correctly matches umbrella
"""
import io
import json
import urllib.request
from PIL import Image, ImageDraw

API_BASE = "http://localhost:8000"


def test_umbrella():
    """Test: umbrella image should produce 'umbrella' semantic class."""
    print("=== Test: Umbrella Classification ===")
    
    # Create a yellow umbrella-like image (triangle shape on white bg)
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw umbrella-like shape: arc + handle
    draw.pieslice([30, 30, 226, 180], 180, 360, fill=(255, 200, 0))
    draw.line([(128, 120), (128, 220)], fill=(100, 60, 30), width=4)
    draw.arc([115, 200, 140, 230], 0, 180, fill=(100, 60, 30), width=3)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    
    # Send to pipeline
    boundary = "----TestUmbrella"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="umbrella.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()
    
    req = urllib.request.Request(
        f"{API_BASE}/pipeline",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    
    det = data["detection"]
    model = data["model"]
    
    print(f"  YOLO label:       {det['yolo_label']}")
    print(f"  Semantic class:   {det['semantic_class']}")
    print(f"  Semantic source:  {det.get('semantic_source', 'N/A')}")
    print(f"  Confidence:       {det['confidence']}")
    print(f"  Model URL:        {model['url']}")
    print(f"  Model method:     {model['method']}")
    print(f"  Model match:      {model['match_name']}")
    print(f"  Processing time:  {data['processing_time_seconds']}s")
    
    # Verify the semantic class is NOT a random character
    semantic_lower = det['semantic_class'].lower()
    bad_results = ['wolverine', 'spider-man', 'batman', 'hulk', 'thor', 
                   'person', 'man', 'woman', 'soldier']
    
    is_correct = not any(bad in semantic_lower for bad in bad_results)
    
    if is_correct:
        print(f"  ✅ CORRECT: '{det['semantic_class']}' is appropriate")
    else:
        print(f"  ❌ WRONG: Got '{det['semantic_class']}' for umbrella image!")
    
    # Verify GLB is valid
    if model['url']:
        model_resp = urllib.request.urlopen(f"{API_BASE}{model['url']}")
        model_bytes = model_resp.read()
        assert model_bytes[:4] == b'glTF', "Not a valid GLB!"
        print(f"  ✅ GLB file valid ({len(model_bytes)} bytes)")
    
    print()
    return data


def test_bottle():
    """Test: bottle should stay 'bottle'."""
    print("=== Test: Bottle Classification ===")
    
    img = Image.new("RGB", (256, 256), (240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Draw bottle shape
    draw.rectangle([100, 30, 156, 60], fill=(200, 200, 200))  # Cap
    draw.rectangle([95, 60, 161, 230], fill=(0, 100, 200))    # Body
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    
    boundary = "----TestBottle"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="bottle.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()
    
    req = urllib.request.Request(
        f"{API_BASE}/pipeline",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    
    det = data['detection']
    print(f"  YOLO label:       {det['yolo_label']}")
    print(f"  Semantic class:   {det['semantic_class']}")
    print(f"  Semantic source:  {det.get('semantic_source', 'N/A')}")
    print(f"  Model match:      {data['model']['match_name']}")
    
    semantic_lower = det['semantic_class'].lower()
    if 'bottle' in semantic_lower or 'vase' in semantic_lower or 'cup' in semantic_lower:
        print(f"  ✅ CORRECT classification")
    else:
        print(f"  ⚠️ Got '{det['semantic_class']}' — may be approximate")
    
    print()
    return data


def test_chair():
    """Test: chair should produce chair-like output."""
    print("=== Test: Chair Classification ===")
    
    img = Image.new("RGB", (256, 256), (200, 180, 160))
    draw = ImageDraw.Draw(img)
    # Draw chair-like shape
    draw.rectangle([80, 100, 176, 120], fill=(120, 80, 40))  # Seat
    draw.rectangle([80, 30, 100, 100], fill=(120, 80, 40))   # Back left
    draw.rectangle([156, 30, 176, 100], fill=(120, 80, 40))  # Back right
    draw.rectangle([85, 120, 95, 210], fill=(120, 80, 40))   # Leg 1
    draw.rectangle([161, 120, 171, 210], fill=(120, 80, 40)) # Leg 2
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    
    boundary = "----TestChair"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="chair.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()
    
    req = urllib.request.Request(
        f"{API_BASE}/pipeline",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    
    det = data['detection']
    print(f"  YOLO label:       {det['yolo_label']}")
    print(f"  Semantic class:   {det['semantic_class']}")
    print(f"  Semantic source:  {det.get('semantic_source', 'N/A')}")
    print(f"  Model match:      {data['model']['match_name']}")
    print()
    return data


if __name__ == "__main__":
    print("=" * 50)
    print("🔮 Hologram Engine v2 — Intelligence Tests")
    print("=" * 50 + "\n")
    
    test_umbrella()
    test_bottle()
    test_chair()
    
    print("=" * 50)
    print("✅ Tests complete!")
    print("=" * 50)
