from kmeans import *
from kohonen import *
from neuralgas import *

if __name__ == "__main__":
    """ K-średnie
        Dostępne wykresy:
        plotScatter(file_name)
        plotVoronoiDiagram(file_name)
    """
    kmeans = KMeans(input_data = "data-1k.txt", k = 5)
    kmeans.start(iterations = 1000, eps = 0.01)
    kmeans.plotScatter(file_name = "kmeans.png")
    #kmeans.plotVoronoiDiagram(file_name = "kmeans-voronoi.png")




    """ Algorytm Kohonena
        Dostępne wykresy:
        - plotScatter(input_data, file_name)
        - plotUMatrix(file_name)
        - plotRGBImage(file_name --- tylko dla danych colors.txt
    """
    #kohonen = Kohonen(input_data = "colors.txt", neurons_size = (10, 10), learning_rate = 0.5, iterations_num = 1000, normalized_data = True)
    #kohonen.start()
    #kohonen.plotScatter(file_name = "kohonen-scatter.png")
    #kohonen.plotUMatrix(file_name = "kohonen-umatrix.png")
    #kohonen.plotRGBImage(file_name = "kohonen-rgb.png")





    """ Gaz neuronowy
        Dostępne wykresy:
        - plotScatter(file_name)
        - plotVoronoiDiagram(file_name)

        normalized_data = True ===> Dane w kółku
        normalized_data = False ===> Dane w spirali
    """
    #neuralgas = NeuralGas(input_data = "data-1k.txt", neurons_size = (10, 10), learning_rate = 0.5, iterations_num = 1000, normalized_data = True)
    #neuralgas.start()
    #neuralgas.plotScatter(file_name = "neural-gas.png")
    #neuralgas.plotVoronoiDiagram(file_name = "neural-gas.png")