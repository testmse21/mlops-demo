# from fastapi import FastAPI, File, UploadFile
# import torch
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# from model import CatClassifier
# from PIL import Image
# from torchvision import transforms

# app = FastAPI()

# # Initialize the model and load saved weights
# model = CatClassifier()
# model.load_state_dict(torch.load("model.pth"))
# model.eval()  # Set the model to evaluation mode

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = Image.open(file.file)
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#     ])
#     img_tensor = transform(image).unsqueeze(0)
#     output = model(img_tensor)
#     pred = torch.argmax(output, dim=1).item()
#     return {"prediction": "Cat" if pred == 1 else "Not Cat"}

from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import CatClassifier
import torchvision.transforms as transforms
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

model = CatClassifier()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    output = model(tensor)
    prediction = torch.argmax(output, 1).item()
    return {"prediction": "Cat" if prediction == 1 else "Not Cat"}
