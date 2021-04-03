import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression

class GeneticAlgorithm():
    """
    params: 
    """

    def __init__(self, model, c_size, n_features):
        self.model = model
        self.chromesome_size = c_size
        self.number_of_features = n_features


    def population_init(self):
        population = []
        for i in range(self.chromesome_size):
            chromosome = np.ones(self.number_of_features, dtype=np.bool)
            chromosome[:int(0.3 * self.number_of_features)] = False
            np.random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def fitness_score(self, population):
        X, y = self.boston_df
        scores = []
        for chromosome in population:
            logmodel.fit(X_train.iloc[:,chromosome], y_train)
            predictions = logmodel.predict(X_test.iloc[:,chromosome])
            scores.append(accuracy_score(y_test,predictions))
        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)
        return list(scores[inds][::-1]), list(population[inds,:][::-1])

# Load in data
boston_df = load_boston()
X, y = boston_df['data'], boston_df['target']
features = boston_df['feature_names']|

ga = GeneticAlgorithm(model=LogisticRegression(), c_size=5, n_features=3)
ga.population_init()