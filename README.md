# Genetic Algorithm for Feature Selection

## Project Aim

The aim of this project is to implement a genetic algorithm to identify the best features to use in a machine learning model.

## Genetic Algorithm (Brief Explanation)

A genetic algorithm is a type of evolutionary algorithm used for optimisation. The algorithm uses analogs of a genetic representation, fitness, genetic recombination, and mutation. Below are the steps we will implement in this genetic algorithm:

1. **Initialise Population:** Firstly, we generate a population of chromosomes. These are represented at an array of boolen values.
2. **Fitness Function:** A fitness score is assigned to each of the chromosomes, in this example sklearn's metric _accuracy_score_ will be used to determine a chromosomes fitness.
3. **Selection:** Select _n_ number of chromosomes as parents based on their respected fitness scores. The genes of these chromosomes are passed onto the next generation to create a new population.
4. **Crossover:** Parent chromosomes are combined creating a new set of chromosomes, these are then added to the new population.
5. **Mutation:** One or more gene values in the chromosome in the new population set generated are altered. Mutation allows the opportunity for more diverse chromosomes to be generated.

## Licence

No licence assigned.
