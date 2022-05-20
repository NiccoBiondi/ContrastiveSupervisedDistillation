import numpy as np


def pair_wise_squared_dist(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2= np.square(a).sum(axis=1) 
    b2 = np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def pair_wise_cosine_dist(a, b, data_is_normalized=False):
    a = np.asarray(a)
    b = np.asarray(b)

    if not data_is_normalized:
        a_normed = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.asarray(a) / np.where(a_normed==0, 1, a_normed)
        b_normed = np.linalg.norm(b, axis=1, keepdims=True)
        b = np.asarray(b) / np.where(b_normed==0, 1, b_normed)

    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    distances = pair_wise_squared_dist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    distances = pair_wise_cosine_dist(x, y)
    return distances
    
class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold=None, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget    # Gating threshold for cosine distance
        self.samples = {}

    def distance(self, queries, galleries):
        return self._metric(queries, galleries)

def _print_distances(distance_matrix, top_n_indice):
    distances = []
    num_row, num_col = top_n_indice.shape
    for r in range(num_row):
        col = []
        for c in range(num_col):
            col.append(distance_matrix[r, top_n_indice[r,c]])
        distances.append(col)

    return distances


def match_k(top_k, galleries, queries):
    
    metric = NearestNeighborDistanceMetric("cosine")
    distance_matrix = metric.distance(queries, galleries)

    top_k_indice = np.argsort(distance_matrix, axis=1)[:, :top_k]
    top_k_distance = _print_distances(distance_matrix, top_k_indice)

    return top_k_indice, top_k_distance

