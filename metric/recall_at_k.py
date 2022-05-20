import numpy as np
import torch
from torch import Tensor
from avalanche.evaluation import Metric

import random

# a standalone metric implementation
class RecallAtK(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self, k=1):
        self.k = k
        self.recall_at_k = 0

    def update(self, distances_matrix, query_labels, gallery_labels, is_equal_query=True ):
        l = len(distances_matrix)

        match_counter = 0

        for i in range(l):
            pos_sim = distances_matrix[i][gallery_labels == query_labels[i]]
            neg_sim = distances_matrix[i][gallery_labels != query_labels[i]]

            thresh = np.sort(pos_sim)[1] if is_equal_query else np.max(pos_sim)
            if np.sum(neg_sim < thresh) >= self.k:
                match_counter = match_counter
            else:
                match_counter += 1
        self.recall_at_k = float(match_counter) / l

    def update_II(self, distances_matrix, query_labels, gallery_labels, is_equal_query=True ):
        """
        Update metric value
        """
        l = len(distances_matrix)
        tot_relevant = 0
        relevant_in_top_k = 0

        for i in range(l):
            tot_relevant += (np.sum(gallery_labels == query_labels[i]) - 1) if is_equal_query else np.sum(gallery_labels == query_labels[i])
            print("Tot relevant ", tot_relevant)
            top_k_indices = np.argsort(distances_matrix[i])[1:self.k+1] if is_equal_query else np.argsort(distances_matrix[i])[:self.k]
            print("Top k indices ",top_k_indices)
            print(gallery_labels[top_k_indices])
            relevant_in_top_k += np.sum(gallery_labels[top_k_indices] == query_labels[i])
            print("#Relevant in top k ",relevant_in_top_k)

        self.precision_at_k = relevant_in_top_k/(self.k*l)
        self.recall_at_k = relevant_in_top_k/tot_relevant

    def result(self) -> float:
        """
        Emit the metric result
        """
        return self.recall_at_k 

    def reset(self):
        """
        Reset the metric value
        """
        self.recall_at_k = 0


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return
def Recall_at_ks(sim_mat, data='cub', query_ids=None, gallery_ids=None):
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['cub'] = [1, 2, 4, 8, 16, 32]
    ks_dict['car'] = [1, 2, 4, 8, 16, 32]
    ks_dict['flw'] = [1, 2, 4, 8, 16, 32]
    ks_dict['craft'] = [1, 2, 4, 8, 16, 32]
    ks_dict['mnist'] = [1, 2, 4, 8, 16, 32]
    ks_dict['dog'] = [1, 2, 4, 8, 16, 32]
    ks_dict['scene'] = [1, 2, 4, 8, 16, 32]
    ks_dict['oct'] = [1, 2, 4, 8, 16, 32]
    ks_dict['jd'] = [1, 2, 4, 8]
    ks_dict['product'] = [1, 10, 100, 1000]
    ks_dict['shop'] = [1, 10, 20, 30, 40, 50]

    if data is None:
        data = 'cub'
    k_s = ks_dict[data]

    m, n = sim_mat.shape 
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)

    num_max = int(1e6)

    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max

    num_valid = np.zeros(len(k_s)) 
    neg_nums = np.zeros(m)
    for i in range(m):
        x = sim_mat[i]

        pos_max = np.max(x[gallery_ids == query_ids[i]]) 
        neg_num = np.sum(x > pos_max)
        neg_nums[i] = neg_num 

    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp - num_valid[i-1]

    return num_valid / float(m)


