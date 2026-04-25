import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ThreadPoolExecutor

class WeightedRandomForest:
    def __init__(self, n_estimators=1500, features_per_other_cat=15, class_weight='balanced', max_workers=4):
        self.n_estimators = n_estimators
        self.features_per_other_cat = features_per_other_cat
        self.class_weight = class_weight
        self.max_workers = max_workers
        self.trees = []
        self.feature_indices_per_tree = []
        self.classes_ = None

        # Feature definition
        self.mfcc_indices = list(range(294, 434))
        self.other_groups = {
            'chroma': range(0, 252),
            'tonnetz': range(252, 293),
            'spectral': range(434, 504),
            'temporal': range(504, 518)
        }

    def _train_single_tree(self, X, y):
        selected_indices = list(self.mfcc_indices)
        for name, indices in self.other_groups.items():
            feature_count = self.features_per_other_cat
            if len(indices) < self.features_per_other_cat:
                feature_count = len(indices) // 2
            selection = np.random.choice(indices, feature_count, replace=False)
            selected_indices.extend(selection)

        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)

        tree = DecisionTreeClassifier(
            max_features='sqrt',
            class_weight=self.class_weight,
            min_samples_leaf=5
        )
        tree.fit(X[idx][:, selected_indices], y[idx])
        return tree, selected_indices

    def fit(self, X, y):
        self.trees = []
        self.feature_indices_per_tree = []
        self.classes_ = np.unique(y)

        if hasattr(X, "values"):
            X = X.values

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(lambda _: self._train_single_tree(X, y), range(self.n_estimators)))

        for tree, indices in results:
            self.trees.append(tree)
            self.feature_indices_per_tree.append(indices)

    def predict(self, X):
        all_probabilities = np.zeros((X.shape[0], len(self.classes_)))

        for i in range(len(self.trees)):
            X_subset = X[:, self.feature_indices_per_tree[i]]

            tree_probs = self.trees[i].predict_proba(X_subset)

            for local_idx, global_class_val in enumerate(self.trees[i].classes_):

                global_idx = np.where(self.classes_ == global_class_val)[0][0]

                all_probabilities[:, global_idx] += tree_probs[:, local_idx]

        final_idx = np.argmax(all_probabilities, axis=1)
        return self.classes_[final_idx]


    def save(self, filename):
        """Saves the entire model state to a file."""
        model_data = {
            'trees': self.trees,
            'feature_indices_per_tree': self.feature_indices_per_tree,
            'classes_': self.classes_,
            'params': {
                'n_estimators': self.n_estimators,
                'features_per_other_cat': self.features_per_other_cat,
                'class_weight': self.class_weight
            }
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Loads the model state from a file."""
        model_data = joblib.load(filename)
        self.trees = model_data['trees']
        self.feature_indices_per_tree = model_data['feature_indices_per_tree']
        self.classes_ = model_data['classes_']

        # Restore parameters
        params = model_data['params']
        self.n_estimators = params['n_estimators']
        self.features_per_other_cat = params['features_per_other_cat']
        self.class_weight = params['class_weight']
        print(f"Model loaded from {filename}")