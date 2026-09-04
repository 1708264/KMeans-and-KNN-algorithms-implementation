__authors__ = ['1708264','1704523','1589611']
__group__ = 'Team_37'

import numpy as np
import math
import utils
import random
from scipy.spatial.distance import cdist

class KMeans:

    def __init__(self, X, K=1, options=None):
        self.num_iter = 0
        self.labels = np.array([[]],dtype=int)
        self.centroids = np.array([[]],dtype=float)
        self.old_centroids = np.array([[]],dtype=float)
        self.W = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)


    def _init_X(self, X):
        if X.ndim == 2:
            self.X = X
            self.X.astype(float)
        elif X.ndim == 3:
            self.W = X.shape[1]
            self.X = X.reshape([(X.shape[0]*X.shape[1]), 3]).astype(float)


    def _init_options(self, options=None):
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 20
        if 'diffrence' not in options:
            options['diffrence'] = 0 
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  
        self.options = options


    def is_near(self, firstK, pixel):
        for RGB in firstK:
            if np.linalg.norm(RGB - pixel) < self.options['diffrence']:
                return True
        return False


    def _init_centroids(self):
        firstK = []
        i = 0
        if self.options['km_init'] == 'first':
            res, index = np.unique(self.X, axis=0, return_index=True)
            unique = self.X[np.sort(index)]
            while len(firstK) != self.K and i != len(unique):
                if not self.is_near(firstK, unique[i]):
                    firstK.append(unique[i])
                i += 1
            self.centroids = np.array(firstK)
        elif self.options['km_init'] == 'random':
            unique = np.unique(self.X, axis=0)
            while len(firstK) != self.K and i != len(unique):
                randomnum = random.randrange(0, len(unique))
                if not self.is_near(firstK, unique[randomnum]):
                    firstK.append(unique[randomnum])
                i += 1
            self.centroids = np.array(firstK)
        elif self.options['km_init'] == 'diagonal':
            diagonal = self.X[:self.W ** 2:self.W + 1]
            res, index = np.unique(diagonal, axis=0, return_index=True)
            unique = diagonal[np.sort(index)]
            while len(firstK) != self.K and i != len(unique):
                if not self.is_near(firstK, unique[i]):
                    firstK.append(unique[i])
                i += 1
            self.centroids = np.array(firstK)


    def get_labels(self):
        self.labels = np.argmin(distance(self.X, np.array(self.centroids)), axis=1)


    def get_centroids(self):
        new_centroids = np.empty((self.K, 3),dtype=float)
        self.old_centroids = self.centroids
        for k in range(self.K):
            pointsK = self.X[self.labels == k]
            if len(pointsK) == 0:
                new_centroids[k] = self.X[random.randrange(0, len(self.X))]
            else:
                new_centroids[k] = np.mean(pointsK, axis=0)
        self.centroids = np.array(new_centroids)


    # def get_centroids(self):
    #     """
    #     Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
    #     """
    #     new_centroids = np.empty((self.K, 3),dtype=float)
    #     self.old_centroids = self.centroids
    #     for k in range(self.K):
    #         mean = np.empty((80 * self.W, 3),dtype=float)
    #         j = 0
    #         for i, index in enumerate(self.labels):
    #             if index == k:
    #                 mean[j][:] = self.X[i][:]
    #                 j += 1
    #         mean.resize((j, 3))
    #         new_centroids[k][:] = np.mean(mean, axis=0)[:]
    #     self.centroids = np.array(new_centroids)


    def converges(self):
        return np.array_equal(self.centroids, self.old_centroids)


    # def converges(self):
    #     """
    #     Checks if there is a difference between current and old centroids
    #     """
    #     for x1,x2 in zip(self.centroids,self.old_centroids):
    #         if x1[0] != x2[0] or x1[1] != x2[1] or x1[2] != x2[2]:
    #             return False
    #     return True


    def fit(self):
        self.num_iter = 0
        self._init_centroids()
        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges():
                break


    def withinClassDistance(self):
        return np.mean(np.sum((self.X - self.centroids[self.labels]) ** 2, axis=1))
    

    def discriminantFischer(self):
        return self.withinClassDistance()/self.interClassDistance()

    
    # def withinClassDistance(self):
    #     A = 0
    #     for i, labels in enumerate(self.labels):
    #         for j in range(3):
    #             A += (self.X[i][j] - self.centroids[labels][j]) ** 2
    #     return A/self.X.shape[0]
    

    def interClassDistance(self):
        return np.sum(np.mean(self.centroids, axis=1))


    def find_bestK(self, max_K):
        self.K = 1
        self.fit()
        if self.options['fitting'] == 'WCD':
            actual = self.withinClassDistance()
        elif self.options['fitting'] == 'ICD':
            actual = self.interClassDistance()
        elif self.options['fitting'] == 'FD':
            actual = self.discriminantFischer()
        for K in range(2, max_K):
            old_centroids = self.centroids
            anterior = actual
            self.K = K
            self.fit()
            if self.options['fitting'] == 'WCD':
                actual = self.withinClassDistance()
            elif self.options['fitting'] == 'ICD':
                actual = self.interClassDistance()
            elif self.options['fitting'] == 'FD':
                actual = self.discriminantFischer()
            if anterior != 0:
                improvement = (anterior - actual) / anterior * 100
            if improvement < self.options['tolerance']:
                self.K -= 1
                self.centroids = old_centroids
                break
        
        
def distance(X, C):
    return cdist(X, C)


# def distance(X, C):

#     A = np.zeros([X.shape[0], C.shape[0]])
#     for i in range(X.shape[0]):
#         for j in range(C.shape[0]):
#             A[i][j] = math.sqrt((X[i][0] - C[j][0]) ** 2 + (X[i][1] - C[j][1]) ** 2 + (X[i][2] - C[j][2]) ** 2)
#     return A


def get_colors(centroids):
    colors = []
    for numeros in utils.get_color_prob(centroids):
        colors.append(utils.colors[np.argmax(numeros)])
    return colors
