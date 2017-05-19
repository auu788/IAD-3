import numpy as np
import matplotlib.pyplot as plt

class Kohonen:
    def __init__(self, input_data, neurons_size, learning_rate, iterations_num):
        self.neurons_size = neurons_size
        self.start_learning_rate = learning_rate
        self.learning_rate = self.start_learning_rate
        self.input_data = input_data
        self.t = 0
        self.max_iterations_num = iterations_num

        self.neurons = [[np.random.random_sample((self.input_data.shape[1],)) for x in range(neurons_size[1])] for y in range(neurons_size[0])]
        self.width = neurons_size[0]
        self.height = neurons_size[1]
        self.map_radius = max(self.width, self.height) / 2.0
        self.TIME = self.max_iterations_num / np.log(self.map_radius)

    def getEuclideanDist(self, a, b):
        return np.linalg.norm(a - b)

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

        self.x = x
        self.y = y
        return winning_neuron

    def getNBHoodRadius(self, iteration):
        return self.map_radius * np.exp(-iteration / self.TIME)

    def getInfluence(self, distance, radius):
        return np.exp(-distance / (2 * (radius * radius)))

    def start(self):
        iteration = 0

        while (iteration < self.max_iterations_num):
            print ("Iteracja: {}".format(iteration))
            nbh_radius = self.getNBHoodRadius(iteration)

            rand_index = np.random.randint(self.input_data.shape[0])
            inp = self.input_data[rand_index]

            #for inp in self.input_data:
            bmu = self.getBMU(inp)
            #print ("{} --- {} --- [{}, {}]".format(inp, bmu, self.x, self.y))
            # Szukanie X i Y dla BMU
            xstart = int(self.x - nbh_radius - 1)
            ystart = int(self.y - nbh_radius - 1)
            xend = int(xstart + (nbh_radius * 2) + 1)
            yend = int(ystart + (nbh_radius * 2) + 1)


            if xend > self.width:
                xend = self.width
            if xstart < 0:
                xstart = 0
            if yend > self.height:
                yend = self.height
            if ystart < 0:
                ystart = 0

            print ("{} - {}".format((xend - xstart), (yend - ystart)))
            for x in range(xstart, xend):
                for y in range(ystart, yend):
                    if self.neurons[x][y] is not bmu:
                        dist = self.getEuclideanDist(bmu, self.neurons[x][y])

                        if dist <= (nbh_radius * nbh_radius):

                            inf = self.getInfluence(dist, nbh_radius)
            #                print ("{}".format((inp - self.neurons[x][y]) * self.learning_rate * inf))
                            self.neurons[x][y] += (inp - self.neurons[x][y]) * self.learning_rate * inf
                #print ("\n")
            #self.createIMG()
            iteration += 1
            self.learning_rate = self.start_learning_rate * np.exp(-iteration / self.max_iterations_num)

    def createIMG(self):
        from PIL import Image

        im = Image.new("RGB", (40, 40))
        pix = im.load()
        for x in range(40):
            for y in range(40):
                r = self.neurons[x][y][0]
                g = self.neurons[x][y][1]
                b = self.neurons[x][y][2]
                rgb = (int(r * 255), int(g * 255), int(b * 255))
                pix[x,y] = rgb

        im.show()




colors = np.array(
         [[0., 0., 1.],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.]])

# store the names of the colors for visualization later on
color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']

with open('data-1k.txt', 'r') as f:
    tmp = f.read().splitlines()
    tmp = [x.split(",") for x in tmp]
    input_data = np.array(tmp, dtype='float64')

kohonen = Kohonen(colors, (40, 40), 0.1, 1000)
kohonen.start()
kohonen.createIMG()
#print (kohonen.neurons)
# for item in kohonen.neurons:
#     plt.scatter(np.array(item)[:, 0], np.array(item)[:, 1], s=50)
#
# plt.show()
