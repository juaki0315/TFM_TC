from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os

from preprocess import preprocess_image
from feature_extractor import extract_features
from model_utils import load_model, predict

app = FastAPI()
model = load_model("mejor_modelo.pkl")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_from_image(file: UploadFile = File(...)):
    temp_path = os.path.join("temp", file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img = preprocess_image(temp_path)
        features = extract_features(img)
        pred = predict(model, features)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_path)

    return JSONResponse(content={"prediction": int(pred)})
