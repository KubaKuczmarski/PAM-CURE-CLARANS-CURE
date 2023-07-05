**[PL]**

# Algorytmy grupowania

Tematem projektu jest implementacja oraz porówanie algorytmów PAM, CLARA, CLARANS oraz CURE dla zadania grupowania.

Zbiór danych - 'Starbucks Locations Worldwide' (plik data.csv)

**Przeprowadzone eksperymenty**
W ramach eksperymentów dokonane zostało porównanie szybkości oraz dokładności działania (miara silhouette) algorytmu CURE oraz CLARANS w zależności od wielkości zbioru danych i liczby klastrów. Dodatkowo zbadany został wpływ sposobu wyznaczania odległości między punktami (odległość euklidesowa lub Manhattan) oraz doboru poszczególnych parametrów algorytmów PAM, CLARA i CLARANS na otrzymane wyniki grupowania.

Eksperymenty dokonano na odpowiednio zmodyfikowanym zbiorze danych „Starbucks Locations Worldwide”. Do grupowania danych wykorzystano ostatnie dwie kolumny - longitude (długość geograficzna) i latitude (szerokość geograficzna) – są to dwie współrzędne używane do określania położenia geograficznego danej restauracji na powierzchni Ziemi. 


**[ENG]**

# Clustering algorithms

The topic of this project is the implementation and comparison of the PAM, CLARA, CLARANS and CURE algorithms for a clustering task.

Dataset - 'Starbucks Locations Worldwide' (data.csv file)

**Experiments carried out
The experiments included a comparison of the speed and accuracy (silhouette measure) of the CURE and CLARANS algorithm as a function of dataset size and number of clusters. In addition, the influence of the method of determining the distance between points (Euclidean or Manhattan distance) and the selection of individual parameters of the PAM, CLARA and CLARANS algorithms on the clustering results obtained was examined.

Experiments were performed on the appropriately modified 'Starbucks Locations Worldwide' dataset. The last two columns, longitude and latitude, were used to group the data - these are the two coordinates used to determine the geographical location of a restaurant on the Earth's surface. 
