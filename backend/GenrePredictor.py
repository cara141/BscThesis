import os

import joblib

from WeightedRandomForest import WeightedRandomForest


class GenrePredictor:
    def __init__(self, genre_name : str):
        self.genre_name = genre_name
        self.scaler = None
        self.encoder = None
        self.model = WeightedRandomForest()

    def load_from_directory(self, base_path):
        folder_path = os.path.join(base_path, self.genre_name)

        if not os.path.exists(folder_path):

            raise FileNotFoundError(f"No directory found for genre: {self.genre_name} in directory {base_path}")

        partial_path = os.path.join(folder_path, self.genre_name)
        self.model.load(partial_path + "_model")
        self.scaler = joblib.load(partial_path + "_scaler")
        self.encoder = joblib.load(partial_path + "_encoder")

    def predict(self, raw_features_518):
        feat_row = raw_features_518.reshape(1,-1)
        scaled_features = self.scaler.transform(feat_row)

        pred_idx = self.model.predict(scaled_features)
        pred_label = self.encoder.inverse_transform(pred_idx)

        return pred_label[0]


