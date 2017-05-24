import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
# pylint: disable=E1101

class Kohonen:
    def __init__(self, input_data, neurons_size, learning_rate, iterations_num):
        with open(input_data, 'r') as f:
            tmp = f.read().splitlines()
            tmp = [x.split(",") for x in tmp]
            input_data = np.array(tmp, dtype='float64')
        
        self.input_data = np.array([(x / np.linalg.norm(x)) for x in input_data]) # Normalizacja do [-1, 1]
        #self.input_data = np.array([((x - np.min(input_data)) / (np.max(input_data) - np.min(input_data))) for x in input_data]) # Normalizacja do [0, 1]

        self.learning_rate = learning_rate
        self.iterations_num = iterations_num
        self.current_iteration = 0
        
        self.neurons = [[np.random.random_sample((self.input_data.shape[1],)) for x in range(neurons_size[1])] for y in range(neurons_size[0])]
        self.width = neurons_size[0]
        self.height = neurons_size[1]

        self.init_learning_rate = self.learning_rate
        self.init_nbhood_radius = max(self.width, self.height) / 2.0
        self.time_const = self.iterations_num / np.log(self.init_nbhood_radius)
        
        self.potential_min = 0.75
        self.potential = np.ones((self.width, self.height))

    def getEuclideanDist(self, a, b):
        return np.linalg.norm(a - b)

    def getNeighborhoodRadius(self, iteration):
        return self.init_nbhood_radius * np.exp(-(iteration / self.time_const))
    
    def getInfluence(self, distance, radius):
        return np.exp(-((distance * distance) / (2 * radius * radius)))
    
    def getNeighbors(self, x, y, radius):
        radius = int(radius) - 1
        rows, cols = self.width, self.height
        neighbors = []

        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if cols > i >= 0 and rows > j >= 0:
                    neighbors.append((i, j))
        
        return neighbors


    def getBMU(self, i_data):
        get_out = False
        for y_i, row in enumerate(self.neurons):
            if get_out:
                break

            for x_i, neuron in enumerate(row):
                if self.potential[y_i][x_i] > self.potential_min:
                    dist = self.getEuclideanDist(i_data, self.neurons[y_i][x_i])
                    winning_neuron = self.neurons[y_i][x_i]
                    x = x_i
                    y = y_i
                    get_out = True
                    break

        for y_i, row in enumerate(self.neurons):
            for x_i, neuron in enumerate(row):
                tmp = self.getEuclideanDist(i_data, neuron)
                if tmp < dist and self.potential[y_i][x_i] > self.potential_min:
                    dist = tmp
                    winning_neuron = neuron
                    x = x_i
                    y = y_i
        
        self.bmu_x = x
        self.bmu_y = y
        self.updatePotential()
        return winning_neuron
    
    def updatePotential(self):
        for y_, row in enumerate(self.neurons):
            for x_, neuron in enumerate(row):
                if y_ == self.bmu_y and x_ == self.bmu_x:
                    self.potential[y_][x_] -= self.potential_min
                else:
                    self.potential[y_][x_] += 1 / (self.width * self.height)
        
        #print (self.bmu_y, self.bmu_x, "\n", self.potential)

    def start(self):
        iter_cnt = 0

        while iter_cnt < self.iterations_num:
            #print ("Iteracja: {}, blad kwantyzacji: {}".format(iter_cnt + 1, self.calculateError()))
            print ("Iteracja: {}".format(iter_cnt + 1))

            rand_index = np.random.randint(self.input_data.shape[0])
            selected_input_data = self.input_data[rand_index]

            bmu = self.getBMU(selected_input_data)
            radius = self.getNeighborhoodRadius(iter_cnt)

            #print (self.bmu_x, self.bmu_y, '\n', self.potential)
            #neighbours = self.getNeighbors(self.bmu_x, self.bmu_y, radius)
            #self.potential[self.bmu_x][self.bmu_y] -= self.pmin

            for y_n, row in enumerate(self.neurons):
                for x_n, neuron in enumerate(row):
                    dist = self.getEuclideanDist(np.array([self.bmu_y, self.bmu_x]), np.array([y_n, x_n]))
                    inf = self.getInfluence(dist, radius)
                    
                    #print ("BMU: [{}, {}], Neuronek: [{}, {}], Dist: {}, Gauss: {}".format(self.bmu_x, self.bmu_y, x_n, y_n, dist, inf))
                    #print ("Przed: ", self.neurons[x_n][y_n])
                    self.neurons[y_n][x_n] += inf * self.learning_rate * (selected_input_data - self.neurons[y_n][x_n])
                    #print ("Po: ", self.neurons[x_n][y_n])
            """for neighbour in neighbours:
                nb_x, nb_y = neighbour[1], neighbour[0]
                dist = self.getEuclideanDist(bmu, self.neurons[nb_x][nb_y])
                inf = self.getInfluence(dist, radius)
                
                self.neurons[nb_x][nb_y] += inf * self.learning_rate * (selected_input_data - self.neurons[nb_x][nb_y])"""

            self.learning_rate = self.init_learning_rate * np.exp(-(iter_cnt / self.time_const))
            iter_cnt += 1
    
    def plotRGBImage(self, file_name):
        from PIL import Image

        im = Image.new("RGB", (self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                r = self.neurons[x][y][0]
                g = self.neurons[x][y][1]
                b = self.neurons[x][y][2]
                rgb = (int(r * 255), int(g * 255), int(b * 255))
                pix[x,y] = rgb

        im.save(file_name)
    
    def calculateError(self):
        error = 0
        for item in self.input_data:
            bmu = self.getBMU(item)
            dist = self.getEuclideanDist(item, bmu)
            error += dist

        return error / len(self.input_data)

    def plotUMatrix(self, file_name):
        outer = []
        for item in self.neurons:
            t = []
            for i in item:
                i[0] += 1
                i[1] -= 1
                t.append(np.mean(i))
            
            outer.append(t)
        
        a = np.array(outer)

        fig = plb.figure()
        plt.imshow(a, interpolation='nearest', cmap=plb.cm.gist_rainbow, extent=(0.5,np.shape(a)[0]+0.5,0.5,np.shape(a)[1]+0.5))
        plt.colorbar()
        plt.savefig(file_name, dpi=700)
    
    def plotScatter(self, file_name):
        out = []
        for row in self.neurons:
            for neuron in row:
                out.append(neuron)
        
        plt.scatter(np.array(self.input_data)[:, 0], np.array(self.input_data)[:, 1], c='b', s=10)

        plt.scatter(np.array(out)[:, 0], np.array(out)[:, 1], c='r', linewidth=1, s=50)
        plt.savefig(file_name, dpi=700)
    
    def plotVoronoiDiagram(self, file_name):
        from scipy.spatial import Voronoi, voronoi_plot_2d

        a = []
        for item in self.neurons:
            for e in item:
                a.append(e)

        vor = Voronoi(a)
        voronoi_plot_2d(vor)
        plt.savefig(file_name, dpi=700)