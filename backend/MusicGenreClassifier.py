import os

from backend.GenrePredictor import GenrePredictor


class MusicGenreClassifier:
    def __init__(self, base_path="../models/"):
        self.base_path = base_path

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

    def predict(self, raw_features, genre=None):
        if genre is None:
            return self.router.predict(raw_features)

        elif genre in self.specialists:
            return self.specialists[genre].predict(raw_features)

        else:
            raise ValueError("Genre not recognized")
