import pickle
import numpy as np
from collections import Counter

def load_classifier(model_path):

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    return clf

def classify_cells(features_df, clf):

    X = features_df.values

    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)

    return predictions, probabilities


def majority_vote(predictions):

    counter = Counter(predictions)

    majority_class = counter.most_common(1)[0][0]
    total_cells = len(predictions)

    confidence = counter[majority_class] / total_cells

    return {
        "final_prediction": majority_class,
        "confidence": confidence,
        "cell_counts": dict(counter)
    }