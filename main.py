from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

from model_utils import load_model
from feature_extractor import preprocess_image, extract_features_from_image

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar modelos una vez
modelo_pipeline = load_model("mejor_modelo.pkl")
final_features = joblib.load("final_features.pkl")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_endpoint(
    file: UploadFile = File(...),
    sex: str = Form(...),
    birthdate: str = Form(...)
):
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Preprocesar imagen y extraer features
        img = preprocess_image(temp_path)
        features = extract_features_from_image(img)
        print("Características extraídas (vector de 512):")
        print(features)

        os.remove(temp_path)

        # Calcular edad
        fecha_nac = pd.to_datetime(birthdate)
        hoy = pd.to_datetime("today")
        edad = (hoy - fecha_nac).days // 365

        # Mapeo de sexo (ajústalo según cómo entrenaste el modelo)
        sexo_mapeado = {"M": 0, "F": 1}
        sexo_transformado = sexo_mapeado.get(sex)

        # Construir dict con features + edad + sexo transformado
        data = {f"feature_{i+1}": features[i] for i in range(len(features))}
        data["edad"] = edad
        data["sex"] = sexo_transformado

        df = pd.DataFrame([data])
        for col in final_features:
            if col not in df.columns:
                df[col] = np.nan
        df = df[final_features]

        print("DataFrame final que se pasa al modelo:")
        print(df.head())
        
        pred = modelo_pipeline.predict(df)[0]
        return {"prediction": int(pred)}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
