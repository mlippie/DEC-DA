import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, calinski_harabasz_score

ari = adjusted_rand_score
homo = homogeneity_score
compl = completeness_score
calihar = calinski_harabasz_score

def nmi(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred, method="arithmetic")

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def get_supervised_metric_handles():
    return [
        ("Normalized mutual information", nmi),
        ("Adjusted rand index", ari),
        ("Homogeneity", homo),
        ("Completeness", compl),
        ("Clustering accuracy", acc)
    ]


def get_unsupervised_metric_handles():
    return [
        ("Calinski Harbasz", calihar)
    ]
