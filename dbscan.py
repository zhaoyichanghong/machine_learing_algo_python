import numpy as np
import matplotlib.pyplot as plt

class Dbscan:
    def fit(self, X, radius, min_points, distance):
        data_number = X.shape[0]
        kernel = set()
        data = set(range(data_number))
        clusters = []

        for i in range(data_number):
            discances = distance(X[i], X)
            if len(np.flatnonzero(discances < radius)) > min_points:
                kernel.add(i)

        while kernel:
            index = np.random.choice(list(kernel), 1)
            cluster_kernel = set(index)
            cluster = set(index)
            data -= cluster

            while cluster_kernel:
                index = np.random.choice(list(cluster_kernel), 1)
                discances = distance(X[index], X)
                neighbor = set(np.flatnonzero(discances < radius))
                delta = neighbor & data
                cluster |= delta
                data -= delta
                cluster_kernel = (cluster_kernel | (kernel & delta)).remove(index[0])

            clusters.append(list(cluster))
            kernel = kernel - cluster
        
        return clusters