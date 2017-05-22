import numpy as np
# pylint: disable=E1101

class NeuralGas:
    def __init__(self, input_data, neurons_size, learning_rate, iterations_num):
        self.input_data = input_data
        self.iterations_num = iterations_num

        self.neurons = [[np.random.random_sample((self.input_data.shape[1],)) for x in range(neurons_size[1])] for y in range(neurons_size[0])]
        self.radius_min = 0.01
        self.radius_max = (neurons_size[0] + neurons_size[1]) / 2.0

        self.learning_rate_min = 0.001
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
    
    def getInfluence(self, position, iteration):
        radius = self.radius_max*pow((self.radius_min / self.radius_max), (iteration / self.iterations_num))
        return np.exp(-(position / radius))

    def start(self):
        for iter_cnt in range(self.iterations_num):
            print (iter_cnt)
            rand_index = np.random.randint(self.input_data.shape[0])
            selected_input_data = self.input_data[rand_index]

            sorted_neurons = self.getSortedNeurons(selected_input_data)
            
            for index, (key, value) in enumerate(sorted_neurons):
                y, x = key[0], key[1]
                inf = self.getInfluence(index, iter_cnt)

                learning_rate = self.learning_rate_init * pow((self.learning_rate_min / self.learning_rate_init), (iter_cnt / self.iterations_num))
                self.neurons[x][y] += learning_rate * inf * (selected_input_data - self.neurons[x][y])
    
    def createIMG(self):
        from PIL import Image
        myk = 10
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
        
colors = np.array(
            [[0., 0., 1.],
            [0., 1., 0.],
            [1., 0., 0.]])

neuralgas = NeuralGas(colors, (10, 10), 0.5, 500)
neuralgas.start()
neuralgas.createIMG()