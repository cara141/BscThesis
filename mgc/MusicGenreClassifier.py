import os

from mgc.FeatureExtractor import FeatureExtractor
from mgc.GenrePredictor import GenrePredictor


class MusicGenreClassifier:
    def __init__(self, base_path="../models/"):
        self.base_path = base_path

        self.extractor = FeatureExtractor()

        self.router = GenrePredictor("Router")
        self.router.load_from_directory(self.base_path)
        self.specialists = {}
        self._load_all_specialists()

    def _load_all_specialists(self):
        for entry in os.listdir(self.base_path):
            full_path = os.path.join(self.base_path, entry)

            if os.path.isdir(full_path) and entry != "Router":
                predictor = GenrePredictor(entry)
                predictor.load_from_directory(self.base_path)

                self.specialists[entry] = predictor

    def predict_audio(self, audio_bytes: bytes, genre_name: str = None):
        feature_vector = self.extractor.extract_from_bytes(audio_bytes)
        return self.predict(feature_vector, genre_name)

    def predict(self, raw_features, genre_name:str = None):
        if genre_name is None:
            return self.router.predict(raw_features)

        elif genre_name in self.specialists:
            return self.specialists[genre_name].predict(raw_features)

        else:
            raise ValueError("Genre not recognized")

    def get_classes(self):
        return list(self.specialists.keys())
