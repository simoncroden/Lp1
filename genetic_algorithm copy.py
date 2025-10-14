import random
import math
from function_data import load_function_data
import numpy as np

# Initialize population:
def initialize_population(M,N,m,n):
    population = []
    for j in range(m):

      instructions = []
      for i in range(n):
        instruction = [
                    random.randint(1,4),   # operator
                    random.randint(1,M),   # destination register
                    random.randint(1,M+N), # operand 1
                    random.randint(1,M+N)  # operand 2
                ]
        instructions.append(instruction)
      population.append(instructions)

    return population





# Decode chromosome:
# DELTE
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

def operators(op1,op2,op):
    if op == 1:
        return op1 + op2
    elif op == 2:
        return op1 - op2
    elif op == 3:
        return op1 * op2
    elif op == 4:
        if op2 != 0:
            return op1 / op2
        else:
            return 0

def decoder(instruction,variable_registers,constant_registers):
    vandc = (variable_registers + constant_registers)
    op1= vandc[instruction[2]-1]
    op2= vandc[instruction[3]-1]
 
    variable_registers[instruction[1]-1] = operators(op1,op2,instruction[0])
    return variable_registers

def evaluate_individual(instructions, variable_registers, constant_registers, data, M=10, N=10):
    
    y_k = []
    for i in data:
        variable_registers = [0] * M
        variable_registers[0] = i[0]  
        

        for instr in instructions:
          variable_registers = decoder(instr, variable_registers, constant_registers)

        y_k.append(variable_registers[0])

    e = 0
    for idx,k in enumerate(data):
        e += (k[1]-y_k[idx])**2
    
    fitness_val = 1 / (np.sqrt(1/len(data)*e))
    return fitness_val


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
    n_genes = len(chromosome1)
    
    pt1, pt2 = sorted(random.sample(range(1, n_genes), 2))
    
    # Ensure minimum segment length (10% of chromosome)
    min_len = max(1, n_genes // 10)
    if pt2 - pt1 < min_len:
        pt2 = min(pt1 + min_len, n_genes - 1)
    
    # Swap the segment between pt1 and pt2
    new1 = [instr.copy() for instr in (chromosome1[:pt1] + chromosome2[pt1:pt2] + chromosome1[pt2:])]
    new2 = [instr.copy() for instr in (chromosome2[:pt1] + chromosome1[pt1:pt2] + chromosome2[pt2:])]
    
    return [new1, new2]

# Mutate individuals:
def mutate(chromosome, M, N, mutation_prob):
    mutated = []
    for instr in chromosome:
        new_instr = instr.copy()
        for field in range(4):
            if random.random() < mutation_prob:
                if field == 0:
                    new_instr[0] = random.randint(1, 4)
                elif field == 1:
                    new_instr[1] = random.randint(1, M)
                else:
                    new_instr[field] = random.randint(1, M + N)
        mutated.append(new_instr)
    return mutated


# Genetic algorithm
def run_function_optimization(
    M, N, m, n, data,
    tournament_size,
    tournament_probability,
    crossover_probability,
    mutation_probability,
    number_of_generations
):
  population = initialize_population(M,N,m,n)
  constant_registers = [random.uniform(-10,10) for t in range(N)]
  stagnation_counter = 0

  for generation_index in range(number_of_generations):
    maximum_fitness = 0
    best_chromosome = []
    fitness_list = []
    for chromosome in population:
      fitness = evaluate_individual(chromosome, [0]*M, constant_registers, data, M, N)
      if (fitness > maximum_fitness):
        maximum_fitness = fitness
        best_chromosome = [instr.copy() for instr in chromosome]
        stagnation_counter = 0
      else:
        stagnation_counter +=1
      fitness_list.append(fitness)

    print(f"Gen {generation_index+1}/{number_of_generations}  gen-best={maximum_fitness:.6g}")  

    temp_population = []
    for i in range(0,m,2):
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

    for i in range(m):
      original_chromosome = temp_population[i]

      current_mutation = min(0.9, mutation_probability * (1 + stagnation_counter / 10.0))
      mutated_chromosome = mutate(original_chromosome, M, N, current_mutation)
      temp_population[i] = mutated_chromosome

    #if random.random() < 0.9:  # 90% chance to keep elite
    temp_population[0] = best_chromosome
    population = temp_population.copy()

    if stagnation_counter >= 10:
            for j in range(50):
                if m - 1 - j >= 2:
                    population[-1 - j] = initialize_population(M, N, 1, n)[0]

  return [maximum_fitness, best_chromosome]
 
data = load_function_data()

M, N = 100, 100
m, n = 300, 300

#print(population)

# Evaluate first individual: evaluate_individual(instructions, variable_registers, constant_registers, data, M=10, N=10):

#for i in range(0,10):
#  print(evaluate_individual(population[i], [0]*M, constant_registers,data, M,N))

best_fitness, best_chromosome = run_function_optimization(
    M, N, m, n, data,
    2,
    0.75,
    0.7,
    0.2,
    1000
)

print(best_fitness,best_chromosome)