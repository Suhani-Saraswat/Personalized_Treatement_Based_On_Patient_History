"""
Native NumPy Decision Tree & Random Forest Classifiers
ABESEC Ghaziabad · Session 2023-24
Project Team: Sneha, Tanisha, Suhani Saraswat | Guide: Dr. Anshul Tickoo

Pure-NumPy implementation designed for high reliability across Windows environments
without external C-extension binary DLL dependencies.
"""

import numpy as np

class PureDecisionTreeClassifier:
    def __init__(self, max_depth=7, min_samples_split=12, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree = None
        self.n_classes_ = 6
        self.feature_importances_ = None
        self.classes_ = np.arange(6)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.n_classes_ = max(6, int(np.max(y)) + 1)
        self.classes_ = np.arange(self.n_classes_)
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        
        self.tree = self._build_node(X, y, depth=0)
        tot = np.sum(self.feature_importances_)
        if tot > 0:
            self.feature_importances_ = self.feature_importances_ / tot
        return self

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / len(y)
        return float(1.0 - np.sum(p ** 2))

    def _build_node(self, X, y, depth):
        n_samples = len(y)
        counts = np.bincount(y, minlength=self.n_classes_)
        proba = counts / max(1, n_samples)
        
        # Base cases
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) <= 1:
            return {"leaf": True, "proba": proba, "class": int(np.argmax(counts))}

        current_gini = self._gini(y)
        best_gain = -1.0
        best_criteria = None

        # Evaluate candidate splits across features
        n_feats = X.shape[1]
        feat_subset = np.random.choice(n_feats, size=min(n_feats, 12), replace=False)
        
        for feat in feat_subset:
            col_vals = X[:, feat]
            # Fast quantile percentiles
            thresholds = np.unique(np.percentile(col_vals, [15, 30, 50, 70, 85]))
            for thresh in thresholds:
                left_mask = col_vals <= thresh
                right_mask = ~left_mask
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left
                
                if n_left < 4 or n_right < 4:
                    continue
                    
                w_left = n_left / n_samples
                w_right = n_right / n_samples
                gain = current_gini - (w_left * self._gini(y[left_mask]) + w_right * self._gini(y[right_mask]))
                
                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (feat, thresh, left_mask, right_mask)

        if best_gain <= 0.0005 or best_criteria is None:
            return {"leaf": True, "proba": proba, "class": int(np.argmax(counts))}

        feat, thresh, left_mask, right_mask = best_criteria
        self.feature_importances_[feat] += best_gain * n_samples

        left_node = self._build_node(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_node(X[right_mask], y[right_mask], depth + 1)
        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": left_node,
            "right": right_node
        }

    def _predict_single(self, node, x):
        if node["leaf"]:
            return node["proba"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_single(node["left"], x)
        return self._predict_single(node["right"], x)

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            return self._predict_single(self.tree, X_arr)
        return np.array([self._predict_single(self.tree, row) for row in X_arr])

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 1:
            return int(np.argmax(proba))
        return np.argmax(proba, axis=1)


class PureRandomForestClassifier:
    def __init__(self, n_estimators=25, max_depth=6, min_samples_split=14, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.classes_ = np.arange(6)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []
        importances = np.zeros(n_features, dtype=float)

        for i in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            
            tree = PureDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i * 37
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            importances += tree.feature_importances_

        tot = np.sum(importances)
        if tot > 0:
            self.feature_importances_ = importances / tot
        else:
            self.feature_importances_ = importances
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        all_probas = [t.predict_proba(X_arr) for t in self.trees]
        return np.mean(all_probas, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 1:
            return int(np.argmax(proba))
        return np.argmax(proba, axis=1)
