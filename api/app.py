from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import os, sys
from pathlib import Path
import torchvision.transforms as transforms

# Make sure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import CatClassifier

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model
model = CatClassifier()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Routes
@app.route("/")
def index():
    # Renders templates/index.html
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image = Image.open(file.stream).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
    prediction = torch.argmax(output, 1).item()
    return jsonify({"prediction": "Cat" if prediction == 1 else "Not Cat"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
