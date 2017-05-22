import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
# pylint: disable=E1101

class NeuralGas:
    def __init__(self, input_data, neurons_size, learning_rate, iterations_num, normalized_data):
        with open(input_data, 'r') as f:
            tmp = f.read().splitlines()
            tmp = [x.split(",") for x in tmp]
            input_data = np.array(tmp, dtype='float64')

        if normalized_data:
            self.input_data = np.array([x / np.linalg.norm(x) for x in input_data])
        else:
            self.input_data = input_data

        self.iterations_num = iterations_num
        self.neurons = [[np.random.random_sample((self.input_data.shape[1],)) for x in range(neurons_size[1])] for y in range(neurons_size[0])]
        self.radius_min = 0.01
        self.radius_max = (neurons_size[0] + neurons_size[1]) / 2

        self.learning_rate_min = 0.01
        self.learning_rate_init = learning_rate

    def getEuclideanDist(self, a, b):
        return np.linalg.norm(a - b)
    
    def getSortedNeurons(self, inp_data):
        distances = {}

        for y, row in enumerate(self.neurons):
            for x, neuron in enumerate(row):

                dist = self.getEuclideanDist(inp_data, neuron)
                distances[(x, y)] = dist
        
        return sorted(distances.items(), key=lambda x:x[1])
    
    def getRadius(self, iteration):
        return self.radius_max * ((self.radius_min / self.radius_max) ** (iteration / self.iterations_num))

    def getInfluence(self, position, radius):
        return np.exp(-(position / radius))

    def start(self):
        for iter_cnt in range(self.iterations_num):
            print ("Iteracja: {}".format(iter_cnt))

            rand_index = np.random.randint(self.input_data.shape[0])
            selected_input_data = self.input_data[rand_index]

            sorted_neurons = self.getSortedNeurons(selected_input_data)
            radius = self.getRadius(iter_cnt)

            for index, (key, value) in enumerate(sorted_neurons):
                y, x = key[0], key[1]
                inf = self.getInfluence(index, radius)
                
                learning_rate = self.learning_rate_init * ((self.learning_rate_min / self.learning_rate_init) ** (iter_cnt / self.iterations_num))
                self.neurons[x][y] += inf * (selected_input_data - self.neurons[x][y])
    
    def plotVoronoiDiagram(self, file_name):
        from scipy.spatial import Voronoi, voronoi_plot_2d

        a = []
        for item in self.neurons:
            for e in item:
                a.append(e)

        vor = Voronoi(a)
        voronoi_plot_2d(vor)
        plt.savefig(file_name, dpi=700)
        #plt.show()
    
    def plotScatter(self, file_name):
        out = []
        for row in self.neurons:
            for neuron in row:
                out.append(neuron)
        
        plt.scatter(np.array(self.input_data)[:, 0], np.array(self.input_data)[:, 1], c='b', s=10)
        plt.scatter(np.array(out)[:, 0], np.array(out)[:, 1], c='r', linewidth=1, s=50)

        #plt.show()
        plt.savefig(file_name, dpi=700)