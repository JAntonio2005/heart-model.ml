from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pathlib import Path
import numpy as np

# Permitir cualquier origen (para pruebas). Ajusta en producción.
origins = ["*"]

app = FastAPI(title="Heart Disease Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo (ruta relativa a la raíz del proyecto)
MODEL_PATH = Path(__file__).resolve().parent / "model" / "heart-disease-v1.joblib"
model = load(MODEL_PATH)

# Debe coincidir con el orden de entrenamiento del CSV
class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class OutputData(BaseModel):
    score: float

@app.get("/")
def root():
    return {"status": "ok", "model_path": str(MODEL_PATH.name)}

@app.post("/score", response_model=OutputData)
def score(data: InputData):
    # Orden exacto según los campos del modelo
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
        data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
        data.ca, data.thal
    ]], dtype=float)

    # Probabilidad de la clase positiva (columna 1)
    proba = model.predict_proba(features)[0, 1]
    return {"score": float(proba)}
