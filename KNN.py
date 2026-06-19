"""
Created on Wed Jun  3 20:08:18 2026

@author: hugonoel
"""

from typing import List, NamedTuple, Tuple, Dict
from collections import Counter, defaultdict
from Vector import Vector, distances
from MachineLearning import split_data
from sklearn.datasets import load_iris
import pandas as pd

class LabeledPoints(NamedTuple):
    point: Vector
    label: str

def majority_vote(labels: List[str]) -> str:
    """Labels doit etre triés du plus proche au plus éloigné"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winner = (len([count
                       for count in vote_counts.values() 
                       if count == winner_count]))
    if num_winner == 1:
        return winner
    else:
        return majority_vote(labels[:-1])
    

def knn_classify(k: int, labeled_points: List[LabeledPoints], new_point: Vector):
    by_distance = sorted(labeled_points, key=lambda lp: distances(lp.point, new_point))
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    return majority_vote(k_nearest_labels)


df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
print(df.columns)

iris_points = [LabeledPoints(
    [row.sepal_length, row.sepal_width, row.petal_length, row.petal_width],
    row.species)
    for row in df.itertuples()]
    
"""
sepal_length = float(input("Entrez la longueur du sepal: "))
sepal_width = float(input("Entrez la largeur du sepal: "))
petal_length = float(input("Entrez la longueur du petal: "))
petal_width = float(input("Entrez la largeur du petal: "))
vect = [sepal_length, sepal_width, petal_length,petal_width]
print(knn_classify(10, iris_points, vect))
"""

iris_train, iris_test = split_data(iris_points, 0.7)

confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0
for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label
    if(predicted == actual):
        num_correct += 1
    confusion_matrix[(predicted, actual)] += 1

print("Pourcentage de réussite", num_correct/len(iris_test))
print(confusion_matrix)
