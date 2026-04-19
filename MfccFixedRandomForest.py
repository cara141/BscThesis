import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier


class MfccFixedRandomForest:
    def __init__(self, n_estimators=1500, features_per_other_cat=15):
        self.n_estimators = n_estimators
        self.features_per_other_cat = features_per_other_cat
        self.trees = []
        self.feature_indices_per_tree = []

        # Chroma: 0-251, Tonnetz: 252-293, MFCC: 294-433, Spectral: 434-503, Temporal: 504-517
        self.mfcc_indices = list(range(294, 434))
        self.other_groups = {
            'chroma': range(0, 252),
            'tonnetz': range(252, 293),
            'spectral': range(434, 504),
            'temporal': range(504, 518)
        }

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            selected_indices = list(self.mfcc_indices)

            for name, indices in self.other_groups.items():
                feature_count = self.features_per_other_cat
                if len(indices) < self.features_per_other_cat:
                    feature_count = len(indices) // 2
                selection = np.random.choice(indices, feature_count, replace=False)
                selected_indices.extend(selection)

            n_samples = X.shape[0]
            idx = np.random.choice(n_samples, n_samples, replace=True)

            tree = DecisionTreeClassifier(max_features='sqrt', class_weight='balanced', min_samples_leaf=5)
            tree.fit(X[idx][:, selected_indices], y[idx])

            self.trees.append(tree)
            self.feature_indices_per_tree.append(selected_indices)

    def predict(self, X):
        tree_predictions = []

        for i, tree in enumerate(self.trees):
            X_subset = X[:, self.feature_indices_per_tree[i]]

            prediction = tree.predict(X_subset)
            tree_predictions.append(prediction)

        final_predictions, count = stats.mode(np.array(tree_predictions), axis=0)

        return final_predictions.ravel()