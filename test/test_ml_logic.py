import pytest
import numpy as np
from mgc.FeatureExtractor import FeatureExtractor
from mgc.MusicGenreClassifier import MusicGenreClassifier
from unittest.mock import MagicMock, patch


@pytest.fixture
def extractor():
    return FeatureExtractor()


def test_feature_extraction_shape(extractor):
    # Create 1 second of white noise (44100 samples)
    fake_audio = np.random.uniform(-1, 1, 44100).astype(np.float32)

    # We need to mock librosa.load because it expects a file/path
    with patch('librosa.load') as mock_load:
        mock_load.return_value = (fake_audio, 44100)

        # Pass dummy bytes
        features = extractor.extract_from_bytes(b"fake_data")

        assert isinstance(features, np.ndarray)
        assert len(features) == 518
        assert features.dtype == np.float32


def test_classifier():
    # Mocking the GenrePredictor to avoid loading heavy physical models during unit tests
    with patch('mgc.GenrePredictor.GenrePredictor') as MockPredictor:
        # Setup mock instances
        mock_router = MockPredictor.return_value
        mock_router.predict.return_value = "Rock"

        # Mock directory structure for specialists
        with patch('os.listdir') as mock_list:
            mock_list.return_value = ["Router", "Rock", "Pop"]
            with patch('os.path.isdir') as mock_isdir:
                mock_isdir.return_value = True

                classifier = MusicGenreClassifier(base_path="C:\\Users\\cezar\\PycharmProjects\\BscThesis\\models")

                fake_features = np.zeros(518)

                try:
                    res = classifier.predict(fake_features)
                except Exception as e:
                    assert False

                assert True


def test_classifier_invalid_genre():
    with patch('mgc.GenrePredictor.GenrePredictor'):
        with patch('os.listdir', return_value=["Router"]):
            classifier = MusicGenreClassifier(base_path="C:\\Users\\cezar\\PycharmProjects\\BscThesis\\models")
            with pytest.raises(ValueError, match="Genre not recognized"):
                classifier.predict(np.zeros(518), genre="NonExistent")