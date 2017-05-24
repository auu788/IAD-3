from kmeans import *
from kohonen import *
from neuralgas import *

if __name__ == "__main__":
    """ K-średnie
        Dostępne wykresy:
        plotScatter(file_name)
        plotVoronoiDiagram(file_name)
    """
    # kmeans = KMeans(input_data = "data-1k.txt", k = 5)
    # kmeans.start(iterations = 1000, eps = 0.01)
    # kmeans.plotScatter(file_name = "kmeans.png")
    #kmeans.plotVoronoiDiagram(file_name = "kmeans-voronoi.png")





    """ Algorytm Kohonena
        Dostępne wykresy:
        - plotScatter(input_data, file_name)
        - plotUMatrix(file_name)
        - plotRGBImage(file_name --- tylko dla danych colors.txt
        - plotVoronoiDiagram(file_name)
    """
    #kohonen = Kohonen(input_data = "data-1k.txt", neurons_size = (30, 30), learning_rate = 0.5, iterations_num = 1000)
    #kohonen.start()
    #print(kohonen.calculateError())
    #kohonen.plotScatter(file_name = "kohonen-scatter.png")
    #kohonen.plotUMatrix(file_name = "kohonen-umatrix.png")
    #kohonen.plotVoronoiDiagram(file_name = "kohenen-voroni.png")
    #kohonen.plotRGBImage(file_name = "kohonen-rgb.png")
    





    """ Gaz neuronowy
        Dostępne wykresy:
        - plotScatter(file_name)
        - plotVoronoiDiagram(file_name)
    """
    #neuralgas = NeuralGas(input_data = "data-1k.txt", neurons_size = (5, 5), learning_rate = 0.5, iterations_num = 100)
    #neuralgas.start()
    #print(neuralgas.calculateError())
    #neuralgas.plotScatter(file_name = "neural-gas.png")
    #neuralgas.plotVoronoiDiagram(file_name = "neural-gas.png")