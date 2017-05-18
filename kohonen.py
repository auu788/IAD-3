import numpy as np

class Node:
    def __init__(self, ileft, iright, itop, ibottom, num_weights):
        self.weights = np.random.random_sample((num_weights,))
        self.x = ileft + (iright - ileft) / 2.0
        self.y = itop + (ibottom - itop) / 2.0

    def getEuclideanDist(self, input_vec):
        return np.linalg.norm(self.weights - input_vec)

class Kohenen:
    def __init__(self, input_data, nodes_num):
        self.input_data = input_data
        self.nodes_num = nodes_num
        self.map_radius = max(constWindowWidth, constWindowHeight)/2
        self.time_constant = iterations_num / np.log(self.map_radius);
        self.neighbourhood_radius = self.map_radius * np.exp(-m_iterations_cnt / self.time_constant);

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

node = Node(1, 4, 3, 1, 5)

#kohenen = Kohenen(input_data, 3)
