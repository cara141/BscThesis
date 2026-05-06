import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from backend.MusicGenreClassifier import MusicGenreClassifier
from backend.FeatureExtractor import FeatureExtractor
from repository.TrackRepository import TrackRepository
from repository.UserRepository import UserRepository
from service.MgcService import MgcService

# App initialization
app = FastAPI()
extractor = FeatureExtractor()
classifier = MusicGenreClassifier()

db_path = "../mgc.db"
user_repo = UserRepository(db_path)
track_repo = TrackRepository(db_path)

service = MgcService(user_repo, track_repo)


# --- Pydantic Schemas ---

class UserRegisterSchema(BaseModel):
    username: str
    email: str
    password: str


class UserLoginSchema(BaseModel):
    username: str
    password: str


class TrackSaveSchema(BaseModel):
    user_id: int
    title: str
    main_genre: str
    sub_genre: str
    features: List[float]


class FeatureInput(BaseModel):
    features: List[float]
    label: str


# --- Classifier Endpoints ---

@app.post("/classifier/audio")
async def predict_raw(file: UploadFile = File(...)):
    audio_data = await file.read()
    try:
        feature_vector = extractor.extract_from_bytes(audio_data)
        prediction = classifier.predict(feature_vector)
        return {
            "features": feature_vector.tolist(),
            "label": prediction
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.post("/classifier/features")
async def predict_features(data: FeatureInput):
    feature_vector = np.array(data.features)
    try:
        prediction = classifier.predict(feature_vector, genre=data.label)
        return {
            "features": data.features,
            "label": prediction
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.get("/classifier/classes")
async def get_classes():
    return {"classes": classifier.get_classes()}


# --- User Endpoints ---

@app.post("/users/register")
async def register(data: UserRegisterSchema):
    try:
        user = service.register_user(data.username, data.email, data.password)
        return {"message": "User created", "user_id": user.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/users/login")
async def login(data: UserLoginSchema):
    user = service.login(data.username, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user.id, "username": user.username}


# --- Track Management Endpoints ---

@app.post("/tracks")
async def save_track(data: TrackSaveSchema):
    try:
        track = service.add_track(
            data.user_id, data.title, data.main_genre, data.sub_genre, data.features
        )
        return {"status": "success", "track_id": track.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tracks/{user_id}")
async def get_user_history(user_id: int, genre: Optional[str] = None, title: Optional[str] = None):
    """
    Returns history for a user. Supports optional filtering by genre or title via query params.
    Example: /tracks/1?genre=Rock
    """
    try:
        if genre:
            tracks = service.search_by_genre(user_id, genre)
        elif title:
            tracks = service.search_by_title(user_id, title)
        else:
            tracks = service.get_user_history(user_id)

        # Convert Domain objects to dicts for JSON serialization
        return [
            {
                "id": t.id,
                "user_id": t.user_id,
                "title": t.title,
                "main_genre": t.main_genre,
                "sub_genre": t.sub_genre,
                "features": t.features
            } for t in tracks
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/tracks/{user_id}/{track_id}")
async def delete_track(user_id: int, track_id: int):
    try:
        service.remove_track(user_id, track_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))