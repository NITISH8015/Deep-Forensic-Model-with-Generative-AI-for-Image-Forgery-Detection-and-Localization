from flask import (
    Flask, render_template, request, redirect,
    session, send_from_directory, flash
)
from pathlib import Path
from PIL import Image, ImageDraw, ImageChops, ImageEnhance
import numpy as np
import cv2, sqlite3, bcrypt
from skimage.metrics import structural_similarity as ssim
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from utils import generate_heatmap

# ✅ AI IMPORTS
from model import load_cnn_model, predict_image
from unet import load_unet_model, predict_mask
from utils import apply_mask_overlay

# ---------- YOLO FACE ----------
face_model = None
try:
    from ultralytics import YOLO
    face_model = YOLO("yolov8n-face.pt")
    print("✅ YOLO face model loaded")
except:
    print("⚠ YOLO face model not available")

# ---------- APP ----------
app = Flask(__name__)
app.secret_key = "final_secure_key"

BASE = Path(__file__).resolve().parent
UPLOAD = BASE / "uploads"
OUTPUT = BASE / "outputs"
DB = BASE / "users.db"

UPLOAD.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------- LOAD MODELS ----------
cnn_model = load_cnn_model()
unet_model = load_unet_model()

# 🔥 ADD THIS
print("CNN:", cnn_model)
print("UNET:", unet_model)

# ---------- DATABASE ----------
def init_db():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password BLOB
        )
    """)
    con.commit()
    con.close()

init_db()

# ---------- HELPERS ----------
def resize_match(ref, tgt):
    return tgt.resize(ref.size) if ref.size != tgt.size else tgt

def detect_face(img_np):
    if face_model is None:
        return None
    results = face_model(img_np, verbose=False)
    if not results or results[0].boxes is None:
        return None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None
    return tuple(map(int, boxes[0]))

# ---------- ELA ----------
def perform_ela(image_path, quality=90):
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

# ---------- CONFIDENCE ----------
def get_confidence_level(percent, ssim_val, regions):
    if percent > 65 or ssim_val < 0.5 or regions > 15:
        return "High Forgery"
    elif percent > 30 or ssim_val < 0.75 or regions > 5:
        return "Medium Forgery"
    else:
        return "Low Forgery"

# ---------- EXPLANATIONS ----------
def generate_user_explanations(regions, percent, ssim_val, face_detected):
    explanations = []

    if face_detected:
        explanations.append("Face area shows visible editing.")

    if percent > 70:
        explanations.append("Heavy manipulation detected.")
    elif percent > 30:
        explanations.append("Moderate manipulation detected.")
    else:
        explanations.append("Minor changes detected.")

    if ssim_val < 0.5:
        explanations.append("High structural difference.")
    elif ssim_val < 0.8:
        explanations.append("Some structural difference.")
    else:
        explanations.append("Images mostly similar.")

    if regions > 10:
        explanations.append("Edits spread across image.")
    elif regions > 0:
        explanations.append("Few localized edits.")
    else:
        explanations.append("No strong edits detected.")

    return explanations

# ---------- PDF ----------
def generate_pdf(data):
    pdf_path = OUTPUT / "forgery_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    y = 800
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Forgery Detection Report")

    y -= 40
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Regions: {data['regions']}")
    y -= 20
    c.drawString(50, y, f"Pixels: {data['percent']}%")
    y -= 20
    c.drawString(50, y, f"SSIM: {data['ssim']}")
    y -= 20
    c.drawString(50, y, f"Confidence: {data['confidence']}")

    c.save()
    return "forgery_report.pdf"

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html", logged=session.get("user"))

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"].encode()

        con = sqlite3.connect(DB)
        cur = con.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (u,))
        row = cur.fetchone()
        con.close()

        if row and bcrypt.checkpw(p, row[0]):
            session["user"] = u
            return redirect("/detect")

        flash("Invalid login")

    return render_template("login.html")

# ---------- SIGNUP ----------
@app.route("/signup", methods=["POST"])
def signup():
    u = request.form["username"]
    e = request.form["email"]
    p = bcrypt.hashpw(request.form["password"].encode(), bcrypt.gensalt())

    try:
        con = sqlite3.connect(DB)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO users(username,email,password) VALUES(?,?,?)",
            (u, e, p)
        )
        con.commit()
        con.close()
        flash("Signup successful")
    except:
        flash("User already exists")

    return redirect("/login")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/detect")
def detect():
    if not session.get("user"):
        return redirect("/login")
    return render_template("detect.html")

# ---------- DUAL IMAGE ----------
@app.route("/upload", methods=["POST"])
def upload():
    if not session.get("user"):
        return redirect("/login")

    original = Image.open(request.files["original"]).convert("RGB")
    forged = Image.open(request.files["forged"]).convert("RGB")
    forged = resize_match(original, forged)

    og = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    fg = cv2.cvtColor(np.array(forged), cv2.COLOR_RGB2GRAY)

    ssim_score, diff = ssim(og, fg, full=True)
    diff_map = ((1 - diff) * 255).astype("uint8")

    _, thresh = cv2.threshold(diff_map, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    localized = forged.copy()
    draw = ImageDraw.Draw(localized)

    regions = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 400:
            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
            regions += 1

    # ✅ CNN
    prediction = predict_image(cnn_model, forged)

    # ✅ U-NET
    mask = predict_mask(unet_model, forged)
    mask.save(OUTPUT / "mask.png")

    # ✅ OVERLAY
    result_overlay = apply_mask_overlay(forged, mask)
    result_overlay.save(OUTPUT / "overlay.png")

    face_box = detect_face(np.array(forged))
    face_detected = False
    if face_box:
        draw.rectangle(face_box, outline="yellow", width=4)
        face_detected = True

    percent = (np.count_nonzero(thresh) / thresh.size) * 100
    confidence = get_confidence_level(percent, ssim_score, regions)

    explanations = generate_user_explanations(
        regions, percent, ssim_score, face_detected
    )

    original.save(OUTPUT / "original.png")
    forged.save(OUTPUT / "forged.png")
    Image.fromarray(diff_map).save(OUTPUT / "difference.png")
    localized.save(OUTPUT / "localized.png")

    pdf = generate_pdf({
        "regions": regions,
        "percent": round(percent, 2),
        "ssim": round(ssim_score, 4),
        "confidence": confidence
    })

    return render_template(
        "result.html",
        regions=regions,
        percent=round(percent, 2),
        ssim=round(ssim_score, 4),
        confidence=confidence,
        prediction=prediction,
        region_details=explanations,
        pdf=pdf
    )

# ---------- SINGLE IMAGE ----------
@app.route("/single-detect", methods=["POST"])
def single_detect():
    if not session.get("user"):
        return redirect("/login")

    try:
        print("STEP 1: Request received")

        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No selected file"

        path = UPLOAD / file.filename
        file.save(path)

        print("STEP 2: Image saved")

        image = Image.open(path).convert("RGB")

        # ✅ CNN Prediction
        prediction = predict_image(cnn_model, image)
        print("STEP 3: Prediction done")

        # ✅ U-NET + HEATMAP + OVERLAY
        try:
            mask = predict_mask(unet_model, image)
            mask.save(OUTPUT / "mask.png")

            result = apply_mask_overlay(image, mask)
            result.save(OUTPUT / "result.png")

            heatmap = generate_heatmap(image, mask)
            heatmap.save(OUTPUT / "heatmap.png")

            print("STEP 4: Mask + overlay + heatmap done")

        except Exception as e:
            print("Mask error:", e)
            image.save(OUTPUT / "result.png")

        # ✅ ELA PROCESSING
        ela = perform_ela(path)

        ela_np = np.array(ela)

        if len(ela_np.shape) == 3:
            ela_np = cv2.cvtColor(ela_np, cv2.COLOR_BGR2GRAY)

        ela_np = ela_np.astype("uint8")

        ela_np = cv2.GaussianBlur(ela_np, (7,7), 0)
        _, thresh = cv2.threshold(ela_np, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ✅ PIXEL ANALYSIS
        total_pixels = thresh.size
        modified_pixels = np.count_nonzero(thresh)
        percent = (modified_pixels / total_pixels) * 100

        # ✅ DRAW REGIONS
        draw = ImageDraw.Draw(image)

        regions = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if cv2.contourArea(c) > 1000:
                draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
                regions += 1

        # ✅ AI SCORE (after regions)
        forgery_score = min(100, round(percent + regions * 2))

        # ✅ FACE DETECTION
        face_box = detect_face(np.array(image))
        face_detected = False

        if face_box:
            draw.rectangle(face_box, outline="yellow", width=4)
            face_detected = True

        # ✅ CONFIDENCE
        confidence = get_confidence_level(percent, 0.7, regions)

        # ✅ EXPLANATIONS (FINAL CLEAN)
        explanations = []

        if percent > 60:
            explanations.append("High pixel-level manipulation detected.")
        elif percent > 30:
            explanations.append("Moderate pixel inconsistencies found.")
        else:
            explanations.append("Minimal editing detected.")

        if regions > 5:
            explanations.append("Forgery spread across multiple regions.")
        elif regions > 0:
            explanations.append("Forgery localized to specific areas.")
        else:
            explanations.append("No strong forgery regions detected.")

        if face_detected:
            explanations.append("Facial region shows possible tampering.")

        # ✅ SAVE OUTPUTS
        image.save(OUTPUT / "single.png")
        ela.save(OUTPUT / "ela.png")

        print("STEP 5: Files saved")

        return render_template(
            "single_result.html",
            regions=regions,
            percent=round(percent, 2),
            confidence=confidence,
            prediction=prediction,
            region_details=explanations,
            score=min(100, round(percent + regions * 2)),   # 🔥 NEW
        )

    except Exception as e:
        print("❌ FULL ERROR:", e)
        return f"<h2>Error:</h2><p>{str(e)}</p>"
    
# ---------- DOWNLOAD ----------
@app.route("/outputs/<f>")
def outputs(f):
    return send_from_directory(OUTPUT, f)

@app.route("/download-report")
def download_report():
    return send_from_directory(OUTPUT, "forgery_report.pdf", as_attachment=True)

@app.route("/project-doc")
def project_doc():
    return send_from_directory(BASE, "image_forging.pdf")

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)