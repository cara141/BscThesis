import numpy as np
from scipy.stats import stats
from sklearn.tree import DecisionTreeClassifier

class CategoricalRandomForest:
    def __init__(self, n_estimators=100, features_per_category=3):
        self.n_estimators = n_estimators
        self.features_per_category = features_per_category

        self.trees = []
        self.feature_indices_per_tree = []

        self.groups = {
            'chroma': range(0, 252),
            'mfcc': range(252, 392),
            'tonnetz': range(392, 434),
            'spectral': range(434, 504),
            'temporal': range(504, 518)
        }

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # 1 Feature selection (features_per_category*no_of_groups = no_of_features_per_tree)
            selected_indices = []
            for group_name, indices in self.groups.items():
                selection = np.random.choice(indices, self.features_per_category, replace=False)
                selected_indices.extend(selection)

            # 2 Bootstrap the data
            n_samples = X.shape[0]
            bootstrap_idX = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_idX][:, selected_indices]
            y_bootstrap = y[bootstrap_idX]

            # 3 Train the tree
            tree = DecisionTreeClassifier(max_features=None, class_weight='balanced')
            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)
            self.feature_indices_per_tree.append(selected_indices)


    def predict(self, X): # Majority vote
        # Individual predictions
        # Aggregate predictions from all trees
        tree_preds = []
        for i, tree in enumerate(self.trees):
            X_subset = X[:, self.feature_indices_per_tree[i]]
            tree_preds.append(tree.predict(X_subset))

        # Majority vote
        return stats.mode(np.array(tree_preds), axis=0)[0]
