import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, K = 5, maximum_iterations = 100):
        self.K = K
        self.maximum_iteration = maximum_iterations 

        #List of clusters
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        #initializing random centroids
        random_samples_indexes = np.random.choice(self.n_samples, self.K, replace = False)
        self.centroids = [self.X[index] for index in random_samples_indexes] 

        #optimizing
        for _ in range(self.maximum_iteration):
            #updating clusters
            self.clusters = self._create_clusters(self.centroids)
            #updating centroids
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            #checking for convergence
            if self._is_converged(old_centroids, self.centroids):
                break
            
        #returning cluster labels
        return self._get_cluster_labels(self.clusters) 

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for index, sample in enumerate(self.X):
            centroid_index = self._closest_centroid(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_index] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, centroids):
        distances = [euclidean_distance(old_centroids[i], centroids[i]) for i in range(self.K)]
        for each in distances:
            if each != 0:
                return False
        return True

    # def plot(self):
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     for i, index in enumerate(self.clusters):
    #         point = self.X[index].T
    #         ax.scatter(*point)
    #     for point in self.centroids:
    #         ax.scatter(*point, marker="x", color="black", linewidth=2)
    #     plt.show()
    
    def generate_output(self):
        output = ""
        for index, each_cluster in enumerate(self.clusters):
            for each in each_cluster:
                output = output + str(self.X[each][0]) + " " + str(self.X[each][1]) + " " + str(index) + "\n"
        return output


if __name__ == "__main__":
    f = open(sys.argv[2], "r")
    clusters = int(sys.argv[1])
    
    
    X = []
    for x in f:
        a, b = x.split("\t")
        X.append([a, b])
    
    X = np.array(X, dtype=np.int32)

    k = KMeans(K=clusters, maximum_iterations=150)
    k.predict(X)
    output = k.generate_output()
    f = open("out.data", "w")
    f.write(output)
    f.close()
    