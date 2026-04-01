from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- LOAD MODEL ----------
print("Loading AI model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ---------- FUNCTIONS ----------

def describe_image(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)


def analyze_image(img, description):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # ---------- AI DETECTION ----------
    score = 0


    ai_words = ["illustration", "render", "3d", "digital art"]
    if any(word in description.lower() for word in ai_words):
        score += 1

    ai_probability = round((score / 4) * 100, 2)
    real_probability = round(100 - ai_probability, 2)

    if ai_probability > 60:
        label = "Likely AI Generated"
    else:
        label = "Likely Real Image"

    return {
        "ai": {
            "label": label,
            "real_confidence": f"{real_probability}%",
            "ai_confidence": f"{ai_probability}%"
        }
    }


# ---------- ROUTES ----------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)

    # Description from BLIP
    description = describe_image(path)

    # Analysis (from dileep.py logic)
    analysis = analyze_image(img, description)

    result = {
        "description": description,
        "ai": analysis["ai"]
    }

    return jsonify(result)


# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # IMPORTANT for Render
    app.run(host="0.0.0.0", port=port)
