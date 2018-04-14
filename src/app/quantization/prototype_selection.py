import numpy as np
import random

def first_match(data, labels, k):
    unique_labels = set(labels)
    prototypes_indexes = []
    for label in unique_labels:
        indexes, = np.where(labels == label)
        for index in indexes[:k]:
            prototypes_indexes.append(index)
    return prototypes_indexes

def random(data, labels, k):
    unique_labels = set(labels)
    prototypes_indexes = []
    for label in unique_labels:
        indexes, = np.where(labels == label)
        random_indexes = random.sample(range(0, len(indexes) - 1), k)
        for random_index in random_indexes:
            prototypes_indexes.append(indexes[random_index])
    return prototypes_indexes