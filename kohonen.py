import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
# pylint: disable=E1101

class Kohonen:
    def __init__(self, input_data, neurons_size, learning_rate, iterations_num):
        self.input_data = input_data
        self.learning_rate = learning_rate
        self.iterations_num = iterations_num
        self.current_iteration = 0
        
        self.neurons = [[np.random.random_sample((self.input_data.shape[1],)) for x in range(neurons_size[1])] for y in range(neurons_size[0])]
        self.width = neurons_size[0]
        self.height = neurons_size[1]

        self.init_learning_rate = self.learning_rate
        self.init_nbhood_radius = (self.width + self.height) / 2.0
        self.time_const = self.iterations_num / np.log(self.init_nbhood_radius)
    
    def getEuclideanDist(self, a, b):
        return np.linalg.norm(a - b)

    def getNeighborhoodRadius(self, iteration):
        return self.init_nbhood_radius * np.exp(-(iteration / self.time_const))
    
    def getInfluence(self, distance, radius):
        return np.exp(-((distance * distance) / (2 * radius * radius)))
    
    def getNeighbors(self, x, y, radius):
        radius = int(radius) - 1
        rows, cols = self.height, self.width
        neighbors = []

        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if cols > i >= 0 and rows > j >= 0:
                    neighbors.append((i, j))
        
        return neighbors

    def getBMU(self, i_data):
        dist = self.getEuclideanDist(i_data, self.neurons[0][0])
        winning_neuron = self.neurons[0][0]
        x = 0
        y = 0

        for y_i, row in enumerate(self.neurons):
            for x_i, neuron in enumerate(row):
                tmp = self.getEuclideanDist(i_data, neuron)
                if tmp < dist:
                    dist = tmp
                    winning_neuron = neuron
                    x = x_i
                    y = y_i

        self.bmu_x = x
        self.bmu_y = y
        return winning_neuron
    
    def start(self):
        iter_cnt = 0

        while iter_cnt < self.iterations_num:
            print (iter_cnt)
            rand_index = np.random.randint(self.input_data.shape[0])
            selected_input_data = self.input_data[rand_index]

            bmu = self.getBMU(selected_input_data)
            radius = self.getNeighborhoodRadius(iter_cnt)

            neighbours = self.getNeighbors(self.bmu_x, self.bmu_y, radius)
            
            for neighbour in neighbours:
                nb_x, nb_y = neighbour[1], neighbour[0]
                dist = self.getEuclideanDist(bmu, self.neurons[nb_x][nb_y])
                inf = self.getInfluence(dist, radius)
                
                self.neurons[nb_x][nb_y] += inf * self.learning_rate * (selected_input_data - self.neurons[nb_x][nb_y])
            
            self.learning_rate = self.init_learning_rate * np.exp(-(iter_cnt / self.time_const))
            iter_cnt += 1
    
    def createIMG(self):
        from PIL import Image
        myk = 50
        im = Image.new("RGB", (myk, myk))
        pix = im.load()
        for x in range(myk):
            for y in range(myk):
                r = self.neurons[x][y][0]
                g = self.neurons[x][y][1]
                b = self.neurons[x][y][2]
                rgb = (int(r * 255), int(g * 255), int(b * 255))
                pix[x,y] = rgb

        im.show()

    def plotUMatrix(self):
        outer = []
        for item in self.neurons:
            tmp = np.zeros((4,4))
            t = []
            for i in item:
                t.append(np.mean(i))
            
            outer.append(t)
        
        
        a = np.array(outer)

        fig = plb.figure()
        plt.imshow(a, interpolation='gaussian', cmap=plb.cm.gist_rainbow, extent=(0.5,np.shape(a)[0]+0.5,0.5,np.shape(a)[1]+0.5))
        plt.colorbar()
        plt.show()

colors = np.array(
            [[0., 0., 1.],
            [0., 1., 0.],
            [1., 0., 0.]])

with open('data-1k.txt', 'r') as f:
    tmp = f.read().splitlines()
    tmp = [x.split(",") for x in tmp]
    input_data = np.array(tmp, dtype='float64')

kohonen = Kohonen(input_data, (5, 5), 0.03, 100)
kohonen.start()