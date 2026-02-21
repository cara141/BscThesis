import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier


class MfccFixedRandomForest:
    def __init__(self, n_estimators=1500, features_per_other_cat=15):
        self.n_estimators = n_estimators
        self.features_per_other_cat = features_per_other_cat
        self.trees = []
        self.feature_indices_per_tree = []

        # Based on your extraction code:
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
            # 1. Start with ALL MFCC features
            selected_indices = list(self.mfcc_indices)

            # 2. Add random samples from the remaining categories
            for name, indices in self.other_groups.items():
                feature_count = self.features_per_other_cat
                if len(indices) < self.features_per_other_cat:
                    feature_count = len(indices) // 2
                selection = np.random.choice(indices, feature_count, replace=False)
                selected_indices.extend(selection)

            # 3. Standard bootstrap and train
            n_samples = X.shape[0]
            idx = np.random.choice(n_samples, n_samples, replace=True)

            # Train the tree on this specific feature subset
            tree = DecisionTreeClassifier(max_features='sqrt', class_weight='balanced', min_samples_leaf=5)
            tree.fit(X[idx][:, selected_indices], y[idx])

            self.trees.append(tree)
            self.feature_indices_per_tree.append(selected_indices)

    def predict(self, X):
        # List to store the vote from each tree
        tree_predictions = []

        for i, tree in enumerate(self.trees):
            # Extract the specific features this tree knows about
            # This includes the 140 MFCCs + the random 'other' features
            X_subset = X[:, self.feature_indices_per_tree[i]]

            # Get the tree's individual opinion
            prediction = tree.predict(X_subset)
            tree_predictions.append(prediction)

        # Majority Vote: Find the most frequent label across all trees
        # axis=0 calculates the mode across the trees for each sample in X
        final_predictions, count = stats.mode(np.array(tree_predictions), axis=0)

        return final_predictions.ravel()