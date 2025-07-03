from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import pandas as pd
import numpy as np
import joblib
import pydicom

from model_utils import load_model
from feature_extractor import preprocess_dicom_image, extract_features_from_image, calculate_age_from_dicom

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar modelo ML y columnas esperadas
modelo_pipeline = load_model("mejor_modelo.pkl")
final_features = joblib.load("final_features.pkl")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporalmente
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Leer dicom
        dicom = pydicom.dcmread(temp_path)

        # Extraer edad
        birthdate_raw = dicom.get("PatientBirthDate", None)
        edad = calculate_age_from_dicom(birthdate_raw) if birthdate_raw else np.nan

        # Sexo
        sex_raw = dicom.get("PatientSex", "U")
        sexo_mapeado = {"M": 0, "F": 1}
        sexo_transformado = sexo_mapeado.get(sex_raw, np.nan)

        # Preprocesar imagen y extraer features
        img = preprocess_dicom_image(dicom)
        features = extract_features_from_image(img)
        print("🧠 Vector de características extraídas:")
        print(features)

        # Construcción del input final
        data = {f"feature_{i+1}": features[i] for i in range(len(features))}
        data["edad"] = edad
        data["sex"] = sexo_transformado

        df = pd.DataFrame([data])
        for col in final_features:
            if col not in df.columns:
                df[col] = np.nan
        df = df[final_features]

        print("📊 DataFrame final que se pasa al modelo:")
        print(df)

        pred = modelo_pipeline.predict(df)[0]

        os.remove(temp_path)

        return {"prediction": int(pred)}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
