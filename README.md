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

## Results

When applying the genetic algorithm to the breast cancer dataset (via _sklearn.datasets_) I managed to generate an increase in model accuracy (**+1.17%**). The results are as follows:

|                Model                | No. of Features | Model Accuracy |
| :---------------------------------: | :-------------: | :------------: |
|         Logistic Regression         |       30        |     95.32%     |
| Logistic Regression (w/ GA applied) |       22        |     96.49%     |

Not only was the GA able to produce a more accurate model, it was also able to reduce the number of features nessessary. The feature importances are below displaying which features have greater coefficient estimates when determining whether a patient is cancerous or not.

|  Coefficient Estimate   | Model Accuracy |
| :---------------------: | :------------: |
|       mean radius       |    1.153073    |
|      radius error       |    0.016773    |
|     concavity error     |    0.001022    |
|    smoothness error     |   -0.002965    |
|     mean perimeter      |   -0.016440    |
|      mean symmetry      |   -0.023514    |
|       area error        |   -0.047486    |
|     perimeter error     |   -0.062453    |
|     worst perimeter     |   -0.141981    |
|        mean area        |   -0.152506    |
|     mean smoothness     |   -0.188362    |
|   mean concave points   |   -0.189271    |
|     mean concavity      |   -0.267668    |
|    worst smoothness     |   -0.338897    |
| worst fractal dimension |   -0.390271    |
|     worst symmetry      |   -0.499931    |
|    mean compactness     |   -0.568610    |
|     worst concavity     |   -0.672451    |
|  concave points error   |   -0.952628    |
| mean fractal dimension  |   -1.308392    |
|  worst concave points   |   -1.375074    |

## Licence

No licence assigned.
