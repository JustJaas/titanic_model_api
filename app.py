
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('titanic_model.pkl')

app = FastAPI(title="API Prediccion Titanic")

class Pasajero(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: bool
    Embarked_S: bool

@app.get("/")
def home():
    return {"mensaje": "API de prediccion Titanic funcionando"}

@app.post("/predecir")
def predecir(pasajero: Pasajero):
    datos = pd.DataFrame([pasajero.model_dump()])
    prediccion = model.predict(datos)[0]
    probabilidad = model.predict_proba(datos)[0][1]
    return {
        "sobrevive": bool(prediccion),
        "probabilidad_supervivencia": round(float(probabilidad), 3)
    }
