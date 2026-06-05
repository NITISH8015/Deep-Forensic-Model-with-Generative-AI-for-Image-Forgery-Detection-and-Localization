from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ---------- OUTPUT FOLDER ----------
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# ---------- HEATMAP FUNCTION ----------
def generate_heatmap(image, mask):
    try:
        img_np = np.array(image)

        # ✅ Resize mask to match image
        mask = mask.resize(image.size)
        mask_np = np.array(mask)

        # ✅ Smooth mask (VERY IMPORTANT)
        mask_np = cv2.GaussianBlur(mask_np, (7,7), 0)

        # ✅ Normalize mask (0–255)
        mask_np = cv2.normalize(mask_np, None, 0, 255, cv2.NORM_MINMAX)

        # ✅ Apply color map
        heatmap = cv2.applyColorMap(mask_np.astype(np.uint8), cv2.COLORMAP_JET)

        # ✅ Blend with original image
        overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

        return Image.fromarray(overlay)

    except Exception as e:
        print("🔥 Heatmap Error:", e)
        return image

# ---------- ELA (ORIGINAL + IMPROVED) ----------
def perform_ela(image_path, quality=90):
    try:
        temp_path = OUTPUT / "temp.jpg"

        original = Image.open(image_path).convert("RGB")
        original.save(temp_path, "JPEG", quality=quality)

        compressed = Image.open(temp_path)

        ela = ImageChops.difference(original, compressed)

        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema])

        scale = 255.0 / max_diff if max_diff != 0 else 1

        ela = ImageEnhance.Brightness(ela).enhance(scale)
        return ela

    except Exception as e:
        print("ELA Error:", e)
        return Image.open(image_path).convert("RGB")


# ---------- MASK OVERLAY (NEW 🔥) ----------
def apply_mask_overlay(image, mask):
    try:
        img_np = np.array(image)
        mask = mask.resize(image.size)   # 🔥 FIX
        mask_np = np.array(mask)

        # Highlight forged region in RED
        mask_np = cv2.GaussianBlur(mask_np, (7,7), 0)
        img_np[mask_np > 150] = [255, 0, 0]

        return Image.fromarray(img_np)

    except Exception as e:
        print("Overlay Error:", e)
        return image


# ---------- IMAGE VALIDATION (NEW 🔐) ----------
def validate_image(file):
    try:
        filename = file.filename.lower()

        # Allowed formats
        if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
            return False, "Only JPG, JPEG, PNG allowed"

        return True, "Valid file"

    except:
        return False, "Invalid file"


# ---------- RESIZE SAFE ----------
def resize_image(image, size=(128,128)):
    try:
        return image.resize(size)
    except:
        return image


# ---------- PERCENT CALCULATION ----------
def calculate_change_percent(thresh):
    try:
        percent = (np.count_nonzero(thresh) / thresh.size) * 100
        return round(percent, 2)
    except:
        return 0