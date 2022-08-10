import io
from ShipClassifier import ShipClassifier
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import torch

SAVED_MODEL_PATH = './saved_models/classifier_weights.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = ShipClassifier().to(device)
state_dict = torch.load(SAVED_MODEL_PATH, 
                        map_location=torch.device(device))
classifier.load_state_dict(state_dict)
classifier.eval()

app = FastAPI()

@app.get("/ping")
def ping():
    return {"Hello": "World"}

@app.post("/infer")
def infer(file: UploadFile=File(...)):
    content = file.file.read()
    img = Image.open(io.BytesIO(content))
    img = img.convert('RGB')
    return classifier.predict_img(img)