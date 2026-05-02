from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import os

app = Flask(__name__)
CORS(app)

# 🔥 biar ringan di Render
torch.set_num_threads(1)
device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LAND_MODEL_PATH = os.path.join(BASE_DIR, "cnn_kondisi_tanah.pth")
PLANT_MODEL_PATH = os.path.join(BASE_DIR, "cnn_penyakit_tanaman.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== LOAD MODEL FUNCTION =====
def load_model(path):
    checkpoint = torch.load(path, map_location=device)

    class_names = checkpoint.get("class_names", None)
    if class_names is None:
        raise Exception("class_names tidak ditemukan di checkpoint")

    model = models.resnet50(weights=None)

    # 🔥 penting: sesuaikan output
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, class_names

# ===== LOAD MODELS =====
try:
    print("📦 Loading models...")

    land_model, land_classes = load_model(LAND_MODEL_PATH)
    print("✅ Model Lahan siap")

    plant_model, plant_classes = load_model(PLANT_MODEL_PATH)
    print("✅ Model Tanaman siap")

except Exception as e:
    print("❌ ERROR LOAD MODEL:", e)
    land_model, plant_model = None, None

# ===== PREDICT FUNCTION =====
def run_predict(model, class_names, image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = class_names[predicted.item()]
    conf = float(confidence.item() * 100)

    return label, conf

# ===== ROUTES =====

@app.route('/')
def home():
    return "API AKTIF"

@app.route('/predict_land', methods=['POST'])
def predict_land():
    if land_model is None:
        return jsonify({"error": "Model lahan belum siap"})

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    try:
        file = request.files['image']
        image = Image.open(file).convert("RGB")

        label, conf = run_predict(land_model, land_classes, image)

        return jsonify({
            "prediction": label,
            "confidence": round(conf, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_plant', methods=['POST'])
def predict_plant():
    if plant_model is None:
        return jsonify({"error": "Model tanaman belum siap"})

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    try:
        file = request.files['image']
        image = Image.open(file).convert("RGB")

        label, conf = run_predict(plant_model, plant_classes, image)

        return jsonify({
            "prediction": label,
            "confidence": round(conf, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# 🔥 WAJIB untuk Render (JANGAN pakai app.run biasa)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)