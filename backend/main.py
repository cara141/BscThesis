from email.mime import multipart

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from backend.MusicGenreClassifier import MusicGenreClassifier
from backend.FeatureExtractor import FeatureExtractor

# App initialization
app = FastAPI()
extractor = FeatureExtractor()
classifier = MusicGenreClassifier()

class FeatureInput(BaseModel):
    features: list[float]
    label: str

@app.post("/predict/audio")
async def predict_raw(file: UploadFile = File(...)):
    audio_data = await file.read()

    try:
        feature_vector = extractor.extract_from_bytes(audio_data)
        prediction = classifier.predict(feature_vector)

        return{
            "features": feature_vector.tolist(),
            "label": prediction
        }

    except Exception as e:
        return {"error" : f"Prediction failed: {str(e)}"}

@app.post("/predict/features")
async def predict_features(data: FeatureInput):
    feature_vector = data.features
    try:
        prediction = classifier.predict(feature_vector, genre=data.label)
    except Exception as e:
        return {"error" : f"Prediction failed: {str(e)}"}
    return {
        "features": data.features,
        "label": prediction
    }