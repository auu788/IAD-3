import numpy as np

class Kohonen:
    def __init__(self, input_data, neurons_size, learning_rate, sigma):
        self.neurons_size = neurons_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.input_data = input_data

        self.neurons = [[np.random.random_sample((self.input_data.shape[1],)) for x in range(neurons_size[1])] for y in range(neurons_size[0])]

    def getEuclideanDist(self, a, b):
        return np.linalg.norm(a - b)

    def getBMU(self):
        for i_data in self.input_data:
            dist = self.getEuclideanDist(i_data, self.neurons[0][0])
            winning_neuron = self.neurons[0][0]

            for row in self.neurons:
                for neuron in row:
                    tmp = self.getEuclideanDist(i_data, neuron)
                    if tmp < dist:
                        dist = tmp
                        winning_neuron = neuron

            print ("{} ----- {} ----- {}".format(i_data, winning_neuron, dist))




colors = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])

# store the names of the colors for visualization later on
color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']


kohonen = Kohonen(colors, (900, 300), 0.3, 0.3)
kohonen.getBMU()
