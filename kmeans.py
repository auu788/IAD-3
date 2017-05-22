import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

class KMeans:
    def __init__(self, input_data, k):
        self.k = k
        self.centroids = None
        self.clusters = None
        self.global_delta = None

        with open(input_data, 'r') as f:
            tmp = f.read().splitlines()
            tmp = [x.split(",") for x in tmp]
            self.input_data = np.array(tmp, dtype='float64')

    def getEuclideanDist(self, a, b):
        return np.linalg.norm(a - b)

    def getBordersOfCoords(self):
        # Szuka minimalnych i maksymalnych wartości dla każdej osi
        maxmins_gen = np.zeros(shape=(len(self.input_data[0]), 2))

        for i, item in enumerate(self.input_data.T):
            tmp = []
            for index in range(self.input_data.shape[0]):
                tmp.append(item[index])
            maxmins_gen[i] = (min(tmp), max(tmp))

        # dla K = 2
        # [[x_min, x_max]
        # [y_min, y_max]]
        return maxmins_gen

    def generateKCentroids(self):
        borders = self.getBordersOfCoords()
        centroids_tmp = np.zeros(shape=(self.k, len(self.input_data[0])))
        for i in range(self.k):
            #print ("Centroid #", i+1)
            tmp = []
            for item in range(len(self.input_data[0])):
                random = int(np.random.uniform(borders[item][0], borders[item][1]))
                #print ("Os #{}, losowa wartosc: {}".format(item+1, random))
                tmp.append(random)

            centroids_tmp[i] = np.array(tmp)

        self.centroids = centroids_tmp
        print ("Wylosowane centroidy:\n", self.centroids, "\n")

    def moveCentroidToMeanOfCluster(self, clusters):
        d = 0
        for centroid_index, points in clusters.items():
            #print ("Grupa: ", centroid_index)
            mean = np.mean(points, axis=0)
            delta = self.centroids[centroid_index] - mean
            self.centroids[centroid_index] = mean

            d += np.mean(delta)

        self.global_delta = abs(d) / len(clusters)
        self.clusters = clusters

    def swapPointsWithCentroids(self, data):
        clusters = defaultdict(list)

        for k, v in data.items():
            tmp_point = self.input_data[k]
            centroid_index = self.centroids.tolist().index(v.tolist())
            clusters[centroid_index].append(tmp_point)

        clusters = dict(clusters)
        # for item in range(len(clusters)):
        #     print ("Centroid: {}".format(self.centroids[item]))
        #     print (clusters[item])

        # średnia z wektora
        #print (np.mean(clusters[1], axis=0))
        return clusters

    def findNearestCentroids(self):
        x = defaultdict(list)
        #print ("Szukanie najbliższych punktów dla centroidów...")

        for index, item in enumerate(self.input_data):
            mini = self.getEuclideanDist(item, self.centroids[0])
            mini_centroid = self.centroids[0]

            for centr in self.centroids:
                y = self.getEuclideanDist(item, centr)
                if y < mini:
                    mini = y
                    mini_centroid = centr

            x[index] = mini_centroid
        x = dict(x)
        x = self.swapPointsWithCentroids(x)
        self.moveCentroidToMeanOfCluster(x)
        #print (x)

    def start(self, iterations = 1000, eps = 0.01):
        self.generateKCentroids()

        for i in range(iterations):
            self.findNearestCentroids()

            print ("Iteracja: {}, blad globalny: {}".format(i+1, self.global_delta))
            if self.global_delta < eps:
                break

    def plotScatter(self, file_name):
        for i, (k, v) in enumerate(self.clusters.items()):
            cmap = plt.cm.get_cmap("hsv", len(self.centroids)+1)
            plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], c=cmap(k), s=10)

            reshaped = np.reshape(self.centroids[k], (1, self.centroids[k].shape[0]))
            plt.scatter(reshaped[:, 0], reshaped[:, 1], marker='o', facecolors='none', edgecolors=cmap(k), linewidth=5, s=300)

        plt.savefig(file_name, dpi=700)

    def plotVoronoiDiagram(self, file_name):
        vor = Voronoi(self.centroids)
        voronoi_plot_2d(vor)
        plt.savefig(file_name, dpi=700)