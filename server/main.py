import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional

from config import settings
from domain.Track import Track

from mgc.MusicGenreClassifier import MusicGenreClassifier

from repository.TrackRepository import TrackRepository
from repository.UserRepository import UserRepository
from server.FeatureInputDto import FeatureInputDto
from server.TrackSaveDto import TrackSaveDto
from server.TrackUpdateDto import TrackUpdateDto
from server.UserLoginDto import UserLoginDto

from server.UserRegisterDto import UserRegisterDto


from service.MusicLibraryService import MusicLibraryService

# app initialization
app = FastAPI()
classifier = MusicGenreClassifier()

db_path = settings.database_path
user_repo = UserRepository(db_path)
track_repo = TrackRepository(db_path)

service = MusicLibraryService(user_repo, track_repo)

@app.post("/classifier/audio")
async def predict_raw(file: UploadFile = File(...)):
    """
    Returns genre prediction based on raw audio along with the extracted features
    """
    audio_data = await file.read()
    try:
        feature_vector = classifier.extractor.extract_from_bytes(audio_data)
        prediction = classifier.predict(feature_vector, None)
        return {
            "features": feature_vector.tolist(),
            "label": prediction
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.post("/classifier/features")
async def predict_features(data: FeatureInputDto):
    """
    Returns genre prediction based on feature input
    """
    feature_vector = np.array(data.features)
    try:
        prediction = classifier.predict(feature_vector, genre_name=data.label)
        return {
            "features": data.features,
            "label": prediction
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/classifier/classes")
async def get_classes():
    """
    Returns classes available for classification.
    """
    return {"classes": classifier.get_classes()}

@app.post("/users/register")
async def register(data: UserRegisterDto):
    """
    Creates a new user and returns the user info.
    """
    try:
        user = service.register_user(data.username, data.email, data.password)
        return {"message": "User created", "user_id": user.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/users/login")
async def login(data: UserLoginDto):
    """
    Authenticates a user and returns the user info.
    """
    user = service.login(data.username, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user.id, "username": user.username}

@app.post("/tracks")
async def save_track(data: TrackSaveDto):
    """
    Creates a new track entry in the database.
    """
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
            tracks = service.get_user_library(user_id)

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
    """
    Deletes a track from the database.
    """
    try:
        service.remove_track(user_id, track_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/tracks/{user_id}/{track_id}")
async def update_track(data: TrackUpdateDto, track_id: int, user_id: int):
    """
    Updates an existing track's metadata.
    """
    try:
        # Map the schema to the domain object the service expects
        updated_track = Track(
            id=track_id,
            user_id=user_id,
            title=data.title,
            main_genre=data.main_genre,
            sub_genre=data.sub_genre,
            features=data.features
        )

        service.update_track_info(updated_track)
        return {"status": "updated", "track_id": data.id}
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))