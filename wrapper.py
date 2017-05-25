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
    #print (kohonen.getNeurons())
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
    #print (neuralgas.getNeurons())
    #neuralgas.start()
    #print(neuralgas.calculateError())
    #neuralgas.plotScatter(file_name = "neural-gas.png")
    #neuralgas.plotVoronoiDiagram(file_name = "neural-gas.png")






    """ Eksperyment 3
        Kohonen - dane muszą być znormalizowane (w kółku), żeby się to jakoś układało, ale są domyślnie znormalizowane
        Neural gas - dane mogą być znormalizowane lub nie, dla oby dwóch zadziała (jest parametr 'normalized')
        K-Means - aby porównać dane z wikampa z tym jak ułożyły się neurony, wypadałoby znormalizować je dla Kohonena, a dla gazu obojętnie (jest parametr 'normalized')
     """
    kohonen = Kohonen(input_data = "data-1k.txt", neurons_size = (30, 30), learning_rate = 0.5, iterations_num = 1000)
    kohonen.start()
    kohonen_neurons = kohonen.getNeurons() # pobieranie neuronków do zmiennej, żeby użyć ich jako wejścia dla k-średnich

    #neuralgas = NeuralGas(input_data = "data-1k.txt", neurons_size = (20, 20), learning_rate = 0.6, iterations_num = 3000, normalized = False)
    #neuralgas.start()
    #neuralgas_neurons = neuralgas.getNeurons() # pobieranie neuronków do zmiennej, żeby użyć ich jako wejścia dla k-średnich

    kmeans = KMeans(input_data = kohonen_neurons, k = 5) # Neurony?
    #kmeans = KMeans(input_data = "data-1k.txt", k = 5, normalized = False) # Czy dane z pliku? Znormalizowane czy nie?
    kmeans.start(iterations = 1000, eps = 0.01)
    kmeans.plotScatter(file_name = "kmeans_3.png")
    kmeans.plotVoronoiDiagram(file_name = "kmeans_3-voronoi.png")
