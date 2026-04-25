import io
import warnings
import librosa
import numpy as np
from scipy import stats


class FeatureExtractor:
    def __init__(self):
        # We define the 7 stats to ensure sorting is consistent
        self.stats = ['kurtosis', 'max', 'mean', 'median', 'min', 'skew', 'std']

    def _feature_stats(self, name, values, data_dict):
        # Create a sub-dictionary for this feature group (e.g., data['mfcc'])
        data_dict[name] = {}

        # Store each stat inside that sub-dictionary
        data_dict[name]['kurtosis'] = np.atleast_1d(stats.kurtosis(values, axis=1))
        data_dict[name]['max'] = np.atleast_1d(np.max(values, axis=1))
        data_dict[name]['mean'] = np.atleast_1d(np.mean(values, axis=1))
        data_dict[name]['median'] = np.atleast_1d(np.median(values, axis=1))
        data_dict[name]['min'] = np.atleast_1d(np.min(values, axis=1))
        data_dict[name]['skew'] = np.atleast_1d(stats.skew(values, axis=1))
        data_dict[name]['std'] = np.atleast_1d(np.std(values, axis=1))

    def extract_from_bytes(self, audio_bytes):
        warnings.filterwarnings('ignore')

        # Load audio from bytes
        x, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        data = {}

        # --- Extraction Pipeline ---
        # ZCR
        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        self._feature_stats('zcr', f, data)

        # CQT
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        self._feature_stats('chroma_cqt', f, data)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        self._feature_stats('chroma_cens', f, data)
        f = librosa.feature.tonnetz(chroma=f)
        self._feature_stats('tonnetz', f, data)
        del cqt

        # STFT
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
        self._feature_stats('chroma_stft', f, data)

        # Librosa 0.10+ uses 'rms' instead of 'rmse'
        f = librosa.feature.rms(S=stft)
        self._feature_stats('rmse', f, data)

        f = librosa.feature.spectral_centroid(S=stft)
        self._feature_stats('spectral_centroid', f, data)
        f = librosa.feature.spectral_bandwidth(S=stft)
        self._feature_stats('spectral_bandwidth', f, data)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        self._feature_stats('spectral_contrast', f, data)
        f = librosa.feature.spectral_rolloff(S=stft)
        self._feature_stats('spectral_rolloff', f, data)

        # MFCC
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        self._feature_stats('mfcc', f, data)

        # --- Flattening to 518 Features ---
        final_features = []

        # Sort groups alphabetically to match FMA structure
        for group in sorted(data.keys()):
            # Sort stats alphabetically: kurtosis, max, mean, median, min, skew, std
            for stat in sorted(self.stats):
                values = data[group][stat]  # This now works with nested dicts
                final_features.extend(values)

        final_array = np.array(final_features, dtype=np.float32)

        if len(final_array) != 518:
            raise ValueError(f"Feature mismatch! Expected 518, got {len(final_array)}")

        return final_array