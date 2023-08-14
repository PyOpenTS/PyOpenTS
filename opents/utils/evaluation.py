import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class auroc:
    """
    Area Under Receiver Operating Characteristic Curve

    :param prediction_scores: array-like of shape (n_samples, n_classes). The multi-class ROC curve requires
        prediction scores for each class. If not specified, will generate its own prediction scores that assume
        100% confidence in selected prediction.
    :param multi_class: {'ovo', 'ovr'}, default='ovo'
        'ovo' computes the average AUC of all possible pairwise combinations of classes.
        'ovr' Computes the AUC of each class against the rest.
    :return: float representing the area under the ROC curve
    """
    def __init__(self, labels, predictions, multi_class='ovo'):
        self.labels = labels
        self.predictions = predictions
        self.multi_class = multi_class
    def load(self):

        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(self.labels).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(self.predictions).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(self.predictions).reshape(-1, 1))
        return roc_auc_score(true_scores, prediction_scores, multi_class=self.multi_class)