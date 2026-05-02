from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import os

app = Flask(__name__)

device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_kondisi_tanah.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_names))
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ Model Loaded")

@app.route('/')
def home():
    return "API AKTIF"

@app.route('/predict_land', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "no image"})

    file = request.files['image']
    image = Image.open(file).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return jsonify({
        "prediction": class_names[pred.item()],
        "confidence": float(conf.item() * 100)
    })
