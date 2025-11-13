from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# model expects the following numeric fields (order matters internally)
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

bundle = joblib.load('model.joblib')  # {'model':..., 'features': [...]}
model = bundle['model']
features = bundle['features']

@app.get('/')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(payload: WineFeatures):
    x = np.array([[payload.fixed_acidity, payload.volatile_acidity, payload.citric_acid,
                   payload.residual_sugar, payload.chlorides, payload.free_sulfur_dioxide,
                   payload.total_sulfur_dioxide, payload.density, payload.pH,
                   payload.sulphates, payload.alcohol]])
    pred = model.predict(x)
    return {'predicted_quality': float(pred[0])}
