import random
import math

# Initialize population:
def initialize_population(population_size, number_of_genes):
    population = [[random.randint(0,1) for gene_index in range(number_of_genes)] for chromosome_index in range(population_size)]
    return population

# Decode chromosome:
def decode_chromosome(chromosome, number_of_variables, maximum_variable_value):
    
    number_of_genes = len(chromosome)
    n_half = int(number_of_genes/number_of_variables)

    x = []
    for i in range(number_of_variables): 
      x_temp = 0
      for j in range(n_half):
          bit_index = i * n_half + j
          x_temp += chromosome[bit_index] * pow(2,-j-1)
      x_temp = -maximum_variable_value + 2*maximum_variable_value*x_temp/(1-pow(2,-n_half))
      x.append(x_temp)
    return x

# Evaluate indviduals:
def evaluate_individual(x):
    x_1 = x[0]
    x_2 = x[1]
    g = (1.5-x_1+x_1*x_2)**2 + (2.25-x_1+x_1*x_2**2)**2 + (2.625-x_1+x_1*x_2**3)**2
    f = 1/(g+1)
    return f


# Select individuals:
def tournament_select(fitness_list, tournament_selection_parameter, tournament_size):
    population_size = len(fitness_list)

    random_indexs = random.sample(range(population_size), tournament_size)
    random_indexs.sort(key=lambda idx: fitness_list[idx], reverse=True)

    for i in range(tournament_size):
      r = random.random()
      if r < tournament_selection_parameter:
         return random_indexs[i]
      
    return random_indexs[-1]

# Carry out crossover:
def cross(chromosome1, chromosome2):
    number_of_genes = len(chromosome1)
    cross_point = random.randint(1,number_of_genes - 1)
    
    new_chromosome_1 = chromosome1[:cross_point] + chromosome2[cross_point:]
    new_chromosome_2 = chromosome2[:cross_point] + chromosome1[cross_point:]

    return [new_chromosome_1, new_chromosome_2]

# Mutate individuals:
def mutate(chromosome, mutation_probability):
    number_of_genes = len(chromosome)
    mutated_chromosome = chromosome.copy()
    for gene_index in range(number_of_genes):
        r = random.random()
        if r < mutation_probability:
            mutated_chromosome[gene_index] = 1 - chromosome[gene_index]

    return mutated_chromosome


# Genetic algorithm
def run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, \
                              tournament_size, tournament_probability, crossover_probability,\
                              mutation_probability, number_of_generations):
  population = initialize_population(population_size,number_of_genes)

  for generation_index in range(number_of_generations):
    maximum_fitness = 0
    best_chromosome = []
    best_individual = []
    fitness_list = []
    for chromosome in population:
      individual = decode_chromosome(chromosome,number_of_variables,maximum_variable_value)
      fitness = evaluate_individual(individual)
      if (fitness > maximum_fitness):
        maximum_fitness = fitness
        best_chromosome = chromosome.copy()  
        best_individual = individual.copy()
      fitness_list.append(fitness)

    temp_population = []
    for i in range(0,population_size,2):
      index_1 = tournament_select(fitness_list, tournament_probability, tournament_size)
      index_2 = tournament_select(fitness_list, tournament_probability, tournament_size)
      chromosome1 = population[index_1].copy()
      chromosome2 = population[index_2].copy()
      r = random.random()
      if r < crossover_probability:
        [new_chromosome_1, new_chromosome_2] = cross(chromosome1,chromosome2)
        temp_population.append(new_chromosome_1)
        temp_population.append(new_chromosome_2) 
      else:
        temp_population.append(chromosome1)
        temp_population.append(chromosome2)

    for i in range(population_size):
      original_chromosome = temp_population[i]

      mutated_chromosome = mutate(original_chromosome, mutation_probability)
      temp_population[i] = mutated_chromosome

    temp_population[0] = best_chromosome
    population = temp_population.copy()

  return [maximum_fitness, best_individual]
 

