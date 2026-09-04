__authors__ = ['1708264','1704523','1589611']


__group__ = '37'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)


    def _init_train(self, train_data):
        train_data = train_data.astype(float)
        self.train_data = train_data.reshape(train_data.shape[0], -1)


    def get_k_neighbours(self, test_data, k):
        test_data = test_data.astype(float)
        test_data = test_data.reshape(test_data.shape[0], -1)
        distances = cdist(test_data, self.train_data)
        sorted = np.argsort(distances, axis=1)
        nearest_indices = []
        for row in sorted:
            nearest_indices.append(row[:k])
        self.neighbors = self.labels[nearest_indices]

 
    def get_class(self):
        result = np.zeros(self.neighbors.shape[0],dtype=object)
        for i, neighbours in enumerate(self.neighbors):
            dict = {}
            for items in neighbours:
                if items not in dict:
                    dict[items] = 1
                else:
                    dict[items] += 1
            result[i] = max(dict, key=dict.get)
        return result


    def predict(self, test_data, k):
        self.get_k_neighbours(test_data, k)
        return self.get_class()