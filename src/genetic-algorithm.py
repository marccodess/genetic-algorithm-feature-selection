# Load dependancies and packages
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load in data
b_cancer_data = load_breast_cancer()
b_cancer_df = pd.DataFrame(
    b_cancer_data["data"], columns=b_cancer_data["feature_names"]
)
target_var = b_cancer_data["target"]

# Split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    b_cancer_df, target_var, test_size=0.3, random_state=10
)

# training a logistics regression model
model = LogisticRegression(solver="lbfgs", max_iter=5000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Accuracy Prior to GA: {round((accuracy_score(y_test, predictions) * 100), 2)}%")

# defining various steps required for the genetic algorithm
def initilization_of_population(size, n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool)
        chromosome[: int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population):
    scores = []
    for chromosome in population:
        model.fit(X_train.iloc[:, chromosome], y_train)
        predictions = model.predict(X_test.iloc[:, chromosome])
        scores.append(accuracy_score(y_test, predictions))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds, :][::-1])


# def selection(pop_after_fit, n_parents):
#     """
#     - This function allows you to state how many parents you want
#       in the selection process.
#     """
#     population_nextgen = []
#     for i in range(n_parents):
#         population_nextgen.append(pop_after_fit[i])
#     return population_nextgen


def selection(pop_after_fit, n_best, n_rand):
    population_nextgen = []
    for i in range(n_best):
        population_nextgen.append(pop_after_fit[i])
    for i in range(n_rand):
        population_nextgen.append(random.choice(pop_after_fit))
    random.shuffle(population_nextgen)
    return population_nextgen


def crossover(pop_after_sel, cross_rate):
    population_nextgen = []
    for i in range(0, len(pop_after_sel), 2):
        try:
            parent1, parent2 = pop_after_sel[i], pop_after_sel[i + 1]
        except:
            pass
        if np.random.randint(0, 1) <= cross_rate:
            cross_point = random.randint(0, len(parent1) - 2)
            child1 = np.array(list(parent1[:cross_point]) + list(parent2[cross_point:]))
            child2 = np.array(list(parent2[:cross_point]) + list(parent1[cross_point:]))
            population_nextgen.append(child1)
            population_nextgen.append(child2)
    return population_nextgen


def mutation(pop_after_cross, mutation_rate):
    population_nextgen = []
    for i in range(len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        if random.random() < mutation_rate:
            mask = np.random.rand(len(chromosome)) < mutation_rate
            chromosome[mask] = False
        population_nextgen.append(chromosome)
    return population_nextgen


# Run the Genetic Algorithm
best_chromo = []
best_score = [0.001]
population_nextgen = initilization_of_population(size=250, n_feat=30)
for i in range(12):  # n_gen
    print(len(population_nextgen))
    scores, pop_after_fit = fitness_score(population_nextgen)
    print(scores[:2])
    pop_after_sel = selection(
        pop_after_fit,
        n_best=int(len(pop_after_fit) * 0.2),
        n_rand=int(100 - int(len(pop_after_fit) * 0.2)),
    )
    # pop_after_sel = selection(pop_after_fit, n_parents=int(len(pop_after_fit) * .95))
    pop_after_cross = crossover(pop_after_sel, cross_rate=0.8)
    population_nextgen = mutation(pop_after_cross, mutation_rate=0.05)
    if scores[0] > best_score[0]:
        best_score[0] = scores[0]
        best_chromo = pop_after_fit[0]

# Apply the GA output to the new model and evaluate
model.fit(X_train.iloc[:, best_chromo], y_train)
predictions = model.predict(X_test.iloc[:, best_chromo])
print(f"Accuracy After GA: {round((accuracy_score(y_test, predictions) * 100), 2)}%")

# Display feature importance for the new model
coeff_df = (
    pd.DataFrame(X_train.iloc[:, best_chromo].columns)
    .reset_index(drop=True)
    .rename(columns={0: "Features"})
)
coeff_df["Coefficient Estimate"] = pd.Series(model.coef_[0])
coeff_df = coeff_df.sort_values(by="Coefficient Estimate", ascending=False).reset_index(
    drop=True
)
print(coeff_df)