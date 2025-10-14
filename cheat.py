import random
from function_data import load_function_data
import matplotlib.pyplot as plt

def TwoPointCrossover(winner1, winner2, population, crossoverProb):
    new_individuals = []

    r = random.random()
    if r < crossoverProb:
        chromosome1 = population[winner1]['Chromosome']
        chromosome2 = population[winner2]['Chromosome']
        length_chromosome1 = len(chromosome1)
        length_chromosome2 = len(chromosome2)
        chromosome_lengths = [length_chromosome1, length_chromosome2]
        crossover_points = []

        for i in range(2):
            valid = False
            while not valid:
                r1 = random.randint(1, chromosome_lengths[i])
                r2 = random.randint(1, chromosome_lengths[i])
                if r1 % 4 == 0 and r2 % 4 == 0 and r1 != r2:
                    if r1 > r2:
                        r1, r2 = r2, r1
                    crossover_points.append((r1, r2))
                    valid = True

        chromosome1_crossPoint1, chromosome1_crossPoint2 = crossover_points[0]
        chromosome2_crossPoint1, chromosome2_crossPoint2 = crossover_points[1]

        # First offspring
        part1 = chromosome1[:chromosome1_crossPoint1]
        part2 = chromosome2[chromosome2_crossPoint1:chromosome2_crossPoint2]
        part3 = chromosome1[chromosome1_crossPoint2:] if chromosome1_crossPoint2 < length_chromosome1 else []
        new_individual1 = part1 + part2 + part3

        # Second offspring
        part1 = chromosome2[:chromosome2_crossPoint1]
        part2 = chromosome1[chromosome1_crossPoint1:chromosome1_crossPoint2]
        part3 = chromosome2[chromosome2_crossPoint2:] if chromosome2_crossPoint2 < length_chromosome2 else []
        new_individual2 = part1 + part2 + part3

    else:
        new_individual1 = population[winner1]['Chromosome']
        new_individual2 = population[winner2]['Chromosome']

    new_individual1 = {'Chromosome': new_individual1}
    new_individual2 = {'Chromosome': new_individual2}
    new_individuals = [new_individual1, new_individual2]

    return new_individuals

import sympy as sp

def EstimateFunction(chromosome, operatorSet):
    """
    Decodes a chromosome into a symbolic function of x.

    Args:
        chromosome (list[int]): Flattened instruction list of length multiple of 4.
        operatorSet (list[int]): Operator indices, typically [1, 2, 3, 4].
    
    Returns:
        sympy.Expr: Simplified symbolic function of x.
    """
    cMax = 1e7
    constantRegisters = ["1", "3", "-1", "2"]
    variableRegisters = ["x", "0", "0"]
    allRegisters = variableRegisters + constantRegisters
    chromosomeLength = len(chromosome)

    for i in range(0, chromosomeLength, 4):
        operatorIndex = chromosome[i]
        destinationIndex = chromosome[i + 1]
        operand1Index = chromosome[i + 2]
        operand2Index = chromosome[i + 3]

        operator = operatorSet[operatorIndex - 1]  # MATLAB is 1-based
        operand1 = allRegisters[operand1Index - 1]
        operand2 = allRegisters[operand2Index - 1]

        if operator == 1:  # addition
            destination = f"({operand1}+{operand2})"
        elif operator == 2:  # subtraction
            destination = f"({operand1}-{operand2})"
        elif operator == 3:  # multiplication
            destination = f"({operand1}*{operand2})"
        elif operator == 4:  # division
            if operand2 != "0":
                destination = f"({operand1}/{operand2})"
            else:
                destination = f"({cMax})"
        else:
            raise ValueError(f"Unknown operator index: {operator}")

        # Update destination register
        allRegisters[destinationIndex - 1] = destination

    # Convert final register 1 to a symbolic expression
    x = sp.Symbol('x')
    expression = sp.sympify(allRegisters[0])
    functionEstimate = sp.simplify(expression)

    return functionEstimate

import numpy as np
def EvaluateIndividual(chromosome, operatorSet, variableRegisters, constantRegisters,
                       maxChromosomeLength, minChromosomeLength, test=False):
    cMax = 1e7
    functionData = load_function_data()  # returns list of lists
    nPoints = len(functionData)
    chromosomeLength = len(chromosome)

    yApprox = np.zeros(nPoints)
    yTrue = np.zeros(nPoints)
    xPoints = np.zeros(nPoints)

    rootMeanSquareError = 0.0

    for iPoint in range(nPoints):
        x = functionData[iPoint][0]  # fixed
        y = functionData[iPoint][1]  # fixed

        # Set up registers
        variableRegisters[0] = x
        for j in range(1, len(variableRegisters)):
            variableRegisters[j] = 0

        allRegisters = variableRegisters + constantRegisters

        # Decode and execute chromosome
        for iGene in range(0, chromosomeLength, 4):
            operatorIndex = chromosome[iGene]
            destinationIndex = chromosome[iGene + 1]
            operand1Index = chromosome[iGene + 2]
            operand2Index = chromosome[iGene + 3]

            operator = operatorSet[operatorIndex - 1]
            operand1 = allRegisters[operand1Index - 1]
            operand2 = allRegisters[operand2Index - 1]

            if operator == 1:
                destination = operand1 + operand2
            elif operator == 2:
                destination = operand1 - operand2
            elif operator == 3:
                destination = operand1 * operand2
            elif operator == 4:
                destination = operand1 / operand2 if operand2 != 0 else cMax
            else:
                raise ValueError(f"Unknown operator index: {operator}")

            allRegisters[destinationIndex - 1] = destination

        yApprox[iPoint] = allRegisters[0]
        yTrue[iPoint] = y
        xPoints[iPoint] = x
        rootMeanSquareError += (y - yApprox[iPoint]) ** 2

    rootMeanSquareError = np.sqrt(rootMeanSquareError / nPoints)
    fitness = 1 / rootMeanSquareError if rootMeanSquareError != 0 else cMax

    # Penalize too short or too long chromosomes
    if chromosomeLength < minChromosomeLength or chromosomeLength > maxChromosomeLength:
        fitness /= cMax

    if test:
        evaluation = {
            "rootMeanSquareError": rootMeanSquareError,
            "xPoints": xPoints,
            "yApprox": yApprox,
            "yTrue": yTrue
        }
    else:
        evaluation = fitness

    return evaluation


import random

def InitializePopulation(populationSize, maxChromosomeLength, minChromosomeLength,
                         operatorSet, variableRegisters, allRegisters):
    """
    Initializes a population of individuals for genetic programming.

    Each chromosome is a flat list of integers encoding instructions:
        [operator, destination, operand1, operand2, operator, destination, ...]
    
    Args:
        populationSize (int): Number of individuals to generate.
        maxChromosomeLength (int): Maximum chromosome length.
        minChromosomeLength (int): Minimum chromosome length.
        operatorSet (list): List of available operators (indices).
        variableRegisters (list): List of variable registers.
        allRegisters (list): List of all registers (variable + constant).
    
    Returns:
        list[dict]: Population as list of dicts, each with key 'Chromosome'.
    """
    OPERATOR = 0
    DESTINATION = 1
    OPERAND_1 = 2
    OPERAND_2 = 3
    NUMBER_OF_INSTRUCTION_VARIABLES = 4

    population = []
    nOperators = len(operatorSet)
    nRegisters = len(allRegisters)
    nVariableRegisters = len(variableRegisters)

    for _ in range(populationSize):
        # Ensure valid chromosome length (multiple of 4)
        valid = False
        while not valid:
            chromosomeLength = minChromosomeLength + int(random.random() * (maxChromosomeLength - minChromosomeLength + 1))
            if chromosomeLength % NUMBER_OF_INSTRUCTION_VARIABLES == 0:
                valid = True

        nInstructions = chromosomeLength // NUMBER_OF_INSTRUCTION_VARIABLES
        tmpChromosome = [[0] * nInstructions for _ in range(NUMBER_OF_INSTRUCTION_VARIABLES)]

        # Randomly assign operator, destination, operands
        for j in range(nInstructions):
            tmpChromosome[OPERATOR][j] = random.randint(1, nOperators)
            tmpChromosome[DESTINATION][j] = random.randint(1, nVariableRegisters)
            tmpChromosome[OPERAND_1][j] = random.randint(1, nRegisters)
            tmpChromosome[OPERAND_2][j] = random.randint(1, nRegisters)

        # Flatten 2D chromosome structure into 1D sequence
        chromosome = [tmpChromosome[i % NUMBER_OF_INSTRUCTION_VARIABLES][i // NUMBER_OF_INSTRUCTION_VARIABLES]
                      for i in range(chromosomeLength)]

        individual = {'Chromosome': chromosome}
        population.append(individual)

    return population

import random

def Mutate(chromosome, operatorSet, variableRegisters, allRegisters, mutationRate):
    """
    Performs mutation on a chromosome.

    Each chromosome is a flat list of integers representing
    [operator, destination, operand1, operand2, ...] instructions.

    Args:
        chromosome (list[int]): Original chromosome to mutate.
        operatorSet (list): List of available operators (indices).
        variableRegisters (list): List of variable registers.
        allRegisters (list): List of all registers (variable + constant).
        mutationRate (float): Mutation scaling factor.

    Returns:
        dict: Mutated individual {'Chromosome': mutated_chromosome}.
    """
    OPERATOR = 0
    DESTINATION = 1
    OPERAND_1 = 2
    OPERAND_2 = 3
    NUMBER_OF_INSTRUCTION_VARIABLES = 4

    chromosomeLength = len(chromosome)
    mutationProb = (1 / chromosomeLength) * mutationRate

    nOperators = len(operatorSet)
    nVariableRegisters = len(variableRegisters)
    nRegisters = len(allRegisters)

    nInstructions = chromosomeLength // NUMBER_OF_INSTRUCTION_VARIABLES

    # Reshape chromosome into instruction matrix (4 × nInstructions)
    tempMutatedIndividual = [
        chromosome[i::NUMBER_OF_INSTRUCTION_VARIABLES] for i in range(NUMBER_OF_INSTRUCTION_VARIABLES)
    ]

    # Mutate operator genes
    for i in range(nInstructions):
        if random.random() < mutationProb:
            tempMutatedIndividual[OPERATOR][i] = random.randint(1, nOperators)

    # Mutate destination genes
    for i in range(nInstructions):
        if random.random() < mutationProb:
            tempMutatedIndividual[DESTINATION][i] = random.randint(1, nVariableRegisters)

    # Mutate operand genes
    for i in range(nInstructions):
        if random.random() < mutationProb:
            tempMutatedIndividual[OPERAND_1][i] = random.randint(1, nRegisters)
        if random.random() < mutationProb:
            tempMutatedIndividual[OPERAND_2][i] = random.randint(1, nRegisters)

    # Flatten back into 1D chromosome
    mutated_chromosome = [
        tempMutatedIndividual[i % NUMBER_OF_INSTRUCTION_VARIABLES][i // NUMBER_OF_INSTRUCTION_VARIABLES]
        for i in range(chromosomeLength)
    ]

    mutatedIndividual = {'Chromosome': mutated_chromosome}
    return mutatedIndividual

import numpy as np
import random

# --- Load all helper functions you’ve translated ---
# from your_module import (
#     InitializePopulation,
#     EvaluateIndividual,
#     TwoPointCrossover,
#     Mutate,
#     TournamentSelect,   # we'll define below if not already
# )

import random
import numpy as np

def TournamentSelect(fitnessList, tournamentProbability, tournamentSize):
    """
    Probabilistic tournament selection.
    Returns the index (0-based) of the selected individual.
    """
    populationSize = len(fitnessList)

    # Select random participants (1-based indexing in MATLAB → adjust to 0-based)
    tourFitnessAndIndividual = np.zeros((tournamentSize, 2))
    for i in range(tournamentSize):
        iTmp = random.randint(0, populationSize - 1)
        tourFitnessAndIndividual[i, 0] = fitnessList[iTmp]   # FITNESS
        tourFitnessAndIndividual[i, 1] = iTmp                 # INDIVIDUAL index

    # Sort by fitness (descending)
    tourFitnessAndIndividual = tourFitnessAndIndividual[np.argsort(-tourFitnessAndIndividual[:, 0])]

    # Selection process
    while True:
        r = random.random()
        if r < tournamentProbability:
            # Return index of the most fit individual
            selectedIndividualIndex = int(tourFitnessAndIndividual[0, 1])
            return selectedIndividualIndex
        else:
            # Remove the most fit if more than one remains
            if len(tourFitnessAndIndividual) > 1:
                tourFitnessAndIndividual = np.delete(tourFitnessAndIndividual, 0, axis=0)
            else:
                # If only one left, return it
                selectedIndividualIndex = int(tourFitnessAndIndividual[0, 1])
                return selectedIndividualIndex


# --- Parameters ---
nGenerations = 3000
populationSize = 100
maxChromosomeLength = 150
minChromosomeLength = 25
tournamentProbability = 0.75
tournamentSize = 5
crossoverProb = 0.8
mutationRate = 80
mutationDecay = 0.9999

nVariableRegisters = 3
operatorSet = [1, 2, 3, 4]  # +, -, *, /
constantRegisters = [1, 3, -1, 2]
variableRegisters = [0] * nVariableRegisters
allRegisters = variableRegisters + constantRegisters

fitnessList = np.zeros(populationSize)
bestFitness = [0, 0]  # [best value, index]

# --- Initialization ---
population = InitializePopulation(populationSize, maxChromosomeLength, minChromosomeLength,
                                  operatorSet, variableRegisters, allRegisters)

# --- Evolution loop ---
for iGeneration in range(1, nGenerations + 1):

    maxFitness = 0
    for individual in range(populationSize):
        chromosome = population[individual]['Chromosome']
        test = False
        fitnessList[individual] = EvaluateIndividual(chromosome, operatorSet,
                                                     variableRegisters.copy(), constantRegisters,
                                                     maxChromosomeLength, minChromosomeLength, test)
        # Elitism
        if fitnessList[individual] > maxFitness:
            maxFitness = fitnessList[individual]
            bestFitness[0] = maxFitness
            bestFitness[1] = individual

    tempPopulation = []

    # Tournament selection + crossover
    for _ in range(0, populationSize, 2):
        winner1 = TournamentSelect(fitnessList, tournamentProbability, tournamentSize)
        winner2 = TournamentSelect(fitnessList, tournamentProbability, tournamentSize)
        newIndividuals = TwoPointCrossover(winner1, winner2, population, crossoverProb)
        tempPopulation.extend(newIndividuals)

    # Elitism — keep best from old generation
    bestIndividualIndex = bestFitness[1]
    bestIndividual = population[bestIndividualIndex]
    tempPopulation[0] = bestIndividual

    # Mutation (skip elite)
    for i in range(1, populationSize):
        chromosome = tempPopulation[i]['Chromosome']
        mutatedIndividual = Mutate(chromosome, operatorSet, variableRegisters,
                                   allRegisters, mutationRate)
        tempPopulation[i] = mutatedIndividual

    # Decay mutation rate
    if mutationRate > 1:
        mutationRate *= mutationDecay

    # Replace old population
    population = tempPopulation

    # Save best chromosome every 1000 generations
    if iGeneration % 1000 == 0:
        bestChromosome = population[0]['Chromosome']
        with open("BestChromosome.txt", "w") as f:
            f.write(" ".join(map(str, bestChromosome)))
        print(f"Generation {iGeneration}")
        e = 1 / maxFitness if maxFitness != 0 else np.inf
        print(f"RMS: {e:.6f}")

# --- Final save ---
fitnessList_sorted = np.sort(fitnessList)[::-1]
bestChromosome = population[0]['Chromosome']
with open("BestChromosome.txt", "w") as f:
    f.write(" ".join(map(str, bestChromosome)))


# --- Load the best chromosome ---
with open("BestChromosome.txt", "r") as f:
    chromosome = list(map(int, f.read().split()))

# --- Evaluate individual ---
evaluation = EvaluateIndividual(chromosome, operatorSet,
                                variableRegisters.copy(), constantRegisters,
                                maxChromosomeLength, minChromosomeLength,
                                test=True)

rootMeanSquareError = evaluation["rootMeanSquareError"]
rootMeanSquareErrorPercent = rootMeanSquareError * 100
xPoints = evaluation["xPoints"]
yApprox = evaluation["yApprox"]
yTrue = evaluation["yTrue"]

# --- Plot ---
plt.figure(figsize=(8,5))
plt.plot(xPoints, yTrue, "-", linewidth=1.5, label="g(x)")
plt.plot(xPoints, yApprox, "--", linewidth=1.5, label="Approximation of g(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# --- Estimate symbolic function ---
functionEstimate = EstimateFunction(chromosome, operatorSet)
print(f"Estimated function: {functionEstimate}")
print(f"Root mean square error: {rootMeanSquareErrorPercent:.2f}%")