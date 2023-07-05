import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import time
import gc


class Point:

    def __init__(self, id, x, y, c):
        self.id = id
        self.x = x
        self.y = y
        self.c = c
        self.distances = []
        self.potential_new_distances = []

    # odległość między dwoma punktami
    def distance(self, point, distance_type):
        if distance_type == 'manhattan':
            return abs(self.x - point.x) + abs(self.y - point.y)
        elif distance_type == 'euklides':
            return math.sqrt(math.pow((self.x - point.x), 2) + math.pow((self.y - point.y), 2))

    # wyznacza odległości od wszystkich medoidów i zapisuje je
    def calculate_distances(self, medoids, distance_type):
        for m in medoids:
            self.distances.append(self.distance(m, distance_type))

    # wyznacza potencjalną różnicę na liście distances po zamianie medoidu na ten o podanym indeksie
    def cost_difference(self, new_medoid, index, distance_type):
        distances = self.distances.copy()
        distances[index] = self.distance(new_medoid, distance_type)
        self.potential_new_distances = distances.copy()
        return min(distances) - min(self.distances)

    # koszt badanego punktu
    def cost(self):
        return min(self.distances)

    # aktualizuje listę odległości od medoidów na podstawie wartości wyznaczonych w cost_difference
    def update(self):
        self.distances = self.potential_new_distances.copy()
        self.potential_new_distances = None

    # restart dystansu
    def restart_distance(self):
        self.distances = []
        self.potential_new_distances = []

# całkowity koszt grupowania
def cost(points):
    cost = 0
    for p in points:
        cost += p.cost()
    return cost

# wybór nowy medoid (algorytm CLARANS)
def find_neighbour(points, medoids, k):
    index_old = random.randint(0, k - 1)
    id_new = -1
    while id_new == -1 or not all(m.id != points[id_new].id for m in medoids):
        id_new = random.randint(0, len(points) - 1)

    new_medoids = medoids.copy()
    new_medoids[index_old] = points[id_new]

    cost_difference = 0
    for p in points:
        cost_difference += p.cost_difference(points[id_new], index_old, distance_type)

    return new_medoids, cost_difference

# Algorytm PAM
def pam(data_path, k, distance_type):
    points = data_path

    counter = 0
    for p in points:
        p.id = counter
        counter += 1

    medoids = random.sample(range(len(points)), k) # randomowe losowanie medoidów
    current_combination = [m for m in medoids]

    for p in points:
        p.calculate_distances([points[m] for m in medoids], distance_type)
    current_cost = cost(points)

    non_medoids_ids = [p.id for p in points if p.id not in medoids]

    gc_old = gc.isenabled()
    gc.disable()
    start_PAM = time.process_time()

    for m in range(k):
        for n in non_medoids_ids:
            cost_difference = 0
            for i in [p.id for p in points]:
                cost_difference += points[i].cost_difference(points[n], m, distance_type)
            if cost_difference < 0:
                current_cost = current_cost + cost_difference
                current_combination[m] = n
                best_medoids = [points[id] for id in current_combination]
                [p.update() for p in points]

    stop_PAM = time.process_time()
    if gc_old: gc.enable()

    time_PAM = stop_PAM - start_PAM

    print(f"Liczba klastrów: {k}")
    print(f"Czas grupowania: {round(time_PAM,2)} sekund")
    print(f"Koszt grupowania: {round(current_cost,4)}")
    print("Medoidy:")
    for i in range(k):
        medoid = best_medoids[i]
        print([medoid.x, medoid.y])
    print(f"Miara odległości miedzy punktami: {distance_type}")

    return points, best_medoids

# Algorytm CLARA
def clara(data_path, k, distance_type, repeats):

    sample_size = 40 + 2 * k
    points_all = data_path
    best_combination = []
    best_medoids = []
    best_cost = float('inf')

    gc_old = gc.isenabled()
    gc.disable()
    start_CLARA = time.process_time()

    for r in range(repeats):
        if len(points_all) > sample_size:
            points = random.sample(points_all, sample_size)
            counter = 0
            for p in points:
                p.id = counter
                counter += 1
        else:
            points = points_all.copy()

        # Wyczyść listę distances dla każdego punktu
        for p in points_all:
            p.restart_distance()

        medoids = random.sample(range(len(points)), k)
        current_combination = [m for m in medoids]
        for p in points:
                p.calculate_distances([points[m] for m in medoids], distance_type)
        current_cost = cost(points)

        non_medoids_ids = [p.id for p in points if p.id not in medoids]

        for m in range(k):
            for n in non_medoids_ids:
                cost_difference = 0
                for i in [p.id for p in points]:
                    cost_difference += points[i].cost_difference(points[n], m, distance_type)
                if cost_difference < 0:
                    current_cost = current_cost + cost_difference
                    current_combination[m] = n
                    [p.update() for p in points]

        if current_cost < best_cost:
            best_cost = current_cost
            best_combination = current_combination
            best_medoids = [points[id] for id in best_combination]

    stop_CLARA = time.process_time()
    if gc_old: gc.enable()

    time_CLARA = stop_CLARA - start_CLARA

    # Wyczyść listę distances dla każdego punktu
    for p in points_all:
        p.restart_distance()

    for p in points_all:
        p.calculate_distances([points_all[id] for id in best_combination], distance_type)

    final_cost = cost(points_all)

    print(f"Liczba klastrów: {k}")
    print(f"Czas grupowania: {round(time_CLARA,2)} sekund")
    print(f"Koszt grupowania: {round(final_cost,4)}")
    print("Medoidy:")
    for i in range(k):
        medoid = best_medoids[i]
        print([medoid.x, medoid.y])
    print(f"Miara odległości miedzy punktami: {distance_type}")

    return points_all, best_medoids

# Algorytm CLARANS
def clarans(data_path, k, distance_type, numlocal,maxneighbours):
    points = data_path
    i = 1
    mincost = float('inf')
    medoids = points[0: k]
    for p in points:
        p.calculate_distances(medoids, distance_type)
    best_medoids = medoids
    current_cost = cost(points)

    gc_old = gc.isenabled()
    gc.disable()
    start_CLARANS = time.process_time()

    while i <= numlocal:
        j = 1

        while j <= maxneighbours:
            new_medoids, cost_difference = find_neighbour(points, medoids, k)
            new_cost = current_cost + cost_difference
            if current_cost > new_cost:
                medoids = new_medoids
                current_cost = new_cost
                [p.update() for p in points]
                j = 1
            else:
                j += 1

        if current_cost < mincost:
            mincost = current_cost
            best_medoids = medoids

        i += 1

    stop_CLARANS = time.process_time()
    if gc_old: gc.enable()

    time_CLARANS = stop_CLARANS - start_CLARANS

    print(f"Liczba klastrów: {k}")
    print(f"Czas grupowania: {round(time_CLARANS,2)} sekund")
    print(f"Koszt grupowania: {round(current_cost,4)}")
    print("Medoidy:")
    for i in range(k):
        medoid = best_medoids[i]
        print([medoid.x, medoid.y])
    print(f"Miara odległości miedzy punktami: {distance_type}")

    return points, best_medoids

# wyznacza wartość miary "silhouette"
def silhouette_score(points, medoids):
    num_points = len(points)
    silhouette_scores = []

    for point in points:
        cluster = point.c
        a = 0
        b = float('inf')

        for other_point in points:
            if other_point.c == cluster and other_point.id != point.id:
                a += point.distance(other_point, distance_type)

            if other_point.c != cluster:
                distance = point.distance(other_point, distance_type)
                if distance < b:
                    b = distance

        a /= (num_points - 1) if num_points > 1 else 1
        s = (b - a) / max(a, b)
        silhouette_scores.append(s)

    return np.mean(silhouette_scores)

# Ładowanie danych
def load_data(nazwa):
    data = pd.read_csv(nazwa)  # Wczytanie danych z pliku CSV
    points = []
    counter = 0
    X = data.iloc[:, -2:].values  # Wybierz ostatnie dwie kolumny jako zbiór danych X
    y = []  # Tutaj możesz umieścić etykiety, jeśli są dostępne w pliku CSV
    for i in X:
        points.append(Point(counter, i[0], i[1],0))
        counter += 1
    random.shuffle(points)
    return points

# Wizualizacja danych
def visualize_clusters_new(points, medoids):
    x = [p.x for p in points]
    y = [p.y for p in points]
    colors = [p.c for p in points]

    unique_clusters = list(set(colors))
    num_clusters = len(unique_clusters)

    # Generowanie unikalnych kolorów dla klastrów
    cluster_colors = list(mcolors.TABLEAU_COLORS.values())[:num_clusters]

    legend_elements = []
    for cluster, color in zip(unique_clusters, cluster_colors):
        cluster_points_x = [p.x for p in points if p.c == cluster]
        cluster_points_y = [p.y for p in points if p.c == cluster]
        plt.scatter(cluster_points_x, cluster_points_y, s=8, color=color, alpha=0.5)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                          markersize=5, label=f'Cluster {cluster}'))
    # Nanieś medoidy dla danego klastra
    medoid_x = [p.x for p in medoids]
    medoid_y = [p.y for p in medoids]
    plt.scatter(medoid_x, medoid_y, marker='x', color='black', label='Medoids')

    plt.legend(handles=legend_elements)
    plt.show()

# Wczytne dane
data = load_data("directory.csv")

# Wybór algorytmu
print("Podaj nazwę algorytmu z którego chcesz skorzytać (możliwe wybory - PAM/CLARA/CLARANS). Wielkość liter nie ma znaczenia." )
algorithm = input("WYBRANY ALGORYTM: ")
algorithm = algorithm.lower()

# Pętla while wykonuje się, dopóki wprowadzona wartość jest niepoprawna
while algorithm not in ['pam', 'clara', 'clarans']:
    print("Niepoprawna nazwa algorytmu. Wybierz spośród opcji: PAM, CLARA, CLARANS.")
    algorithm = input("WYBRANY ALGORYTM: ")
    algorithm = algorithm.lower()

if algorithm == 'clara':
    print("Zdefiniuj liczbę powtórzeń algorytmu dla próbki losowej.")
    while True:
        try:
            repeats = int(input("LICZBA POWTÓRZEŃ: "))
            break
        except ValueError:
            print("Niepoprawna wartość. Podaj liczbę.")
elif algorithm == 'clarans':
    print("Podaj liczbę przeszukiwanych sąsiadów - maxneighbours i liczbę szukanych minimów - numlocal.")
    while True:
        try:
            maxneighbours = int(input("LICZBA SĄSIADÓW: "))
            numlocal = int(input("LICZBA MINIMÓW: "))
            break
        except ValueError:
            print("Niepoprawna wartość. Podaj liczbę.")

# Liczba klastrów
print("Podaj na ile klastrów chcesz podzielić dane")
while True:
    try:
        k = int(input("LICZBA KLASTRÓW: "))
        break
    except ValueError:
        print("Niepoprawna wartość. Podaj liczbę.")

# Odległość między punktami
print("Podaj nazwę miary odległości między punktami (możliwe wybory - euklides/kosinus/manhattan). Wielkość liter nie ma znaczenia.")
distance_type = input("WYBRANY ODLEGŁOŚCI: ")
distance_type = distance_type.lower()

# Pętla while wykonuje się, dopóki wprowadzona wartość jest niepoprawna
while distance_type not in ['euklides', 'kosinus', 'manhattan']:
    print("Niepoprawna nazwa miary odległości. Wybierz spośród opcji: euklides, kosinus, manhattan.")
    distance_type = input("WYBRANY ODLEGŁOŚCI: ")
    distance_type = distance_type.lower()

print("--------------------------------------------------------------------------------------------------------------------------------" )
print("WYNIKI" )
if algorithm == "clarans":
    pts, bestnode = clarans(data, k, distance_type, numlocal, maxneighbours)
elif algorithm == "clara":
    pts, bestnode = clara(data, k, distance_type, repeats)
else:
    pts, bestnode = pam(data, k, distance_type)

for p in pts:
    p.c = np.argmin(p.distances)

# Miara silhouette
silhouette = silhouette_score(pts, k)
print(f"Miara jakości grupowania - Silhouette: {silhouette}")

# Wizualizacja wyników
visualize_clusters_new(pts, bestnode)

