import numpy as np
import random
from function_data import load_function_data

def tournament_select(fitness_list, tournament_selection_parameter, tournament_size):
    population_size = len(fitness_list)

    random_indexs = random.sample(range(population_size), tournament_size)
    random_indexs.sort(key=lambda idx: fitness_list[idx], reverse=True)

    for i in range(tournament_size):
      r = random.random()
      if r < tournament_selection_parameter:
         return random_indexs[i]
      
    return random_indexs[-1]

def cross(chromosome1, chromosome2):
    number_of_genes = len(chromosome1)
    pt1, pt2 = sorted(random.sample(range(1, number_of_genes), 2))
    
    new_chromosome_1 = (
        chromosome1[:pt1] + chromosome2[pt1:pt2] + chromosome1[pt2:]
    )
    new_chromosome_2 = (
        chromosome2[:pt1] + chromosome1[pt1:pt2] + chromosome2[pt2:]
    )
    return new_chromosome_1, new_chromosome_2

def mutate(instr, M, N):
    """Mutate a single instruction [op, dst, src1, src2]."""
    field = random.randint(0, 3)
    if field == 0:  # mutate operator
        instr[0] = random.randint(1, 4)
    elif field == 1:  # mutate destination register
        instr[1] = random.randint(1, M)
    else:  # mutate operands
        instr[field] = random.randint(1, M+N)
    return instr

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

M = 100
N = 100
variable_registers = [1] * M
constant_registers = [random.randint(1, 1000) for _ in range(N)]

data = load_function_data()

#print(instructions)
#print(decoder(instructions[0],variable_registers,constant_registers))

def fitness(instructions, data, M=10, N=10):

    y_k = []
    for i in data:
        variable_registers = [0] * M
        variable_registers[0] = i[0]  
        
        for instr in instructions:
            decoder(instr, variable_registers, constant_registers)
        y_k.append(variable_registers[0])

    e = 0
    for idx,k in enumerate(data):
        e += (k[1]-y_k[idx])**2
    
    fitness_val = 1 / (np.sqrt(1/len(data)*e))
    return fitness_val

def ga(pop_size=20, gens=50, prog_len=10, M=10, N=10):
    data = load_function_data()
    
    # Initialize population of programs
    population = []
    for _ in range(pop_size):
        program = []
        for _ in range(prog_len):
            instr = [
                random.randint(1,4),   # operator
                random.randint(1,M),   # destination register
                random.randint(1,M+N), # operand 1
                random.randint(1,M+N)  # operand 2
            ]
            program.append(instr)
        population.append(program)
    
    # evolution loop
    for g in range(gens):
        fitnesses = [fitness(prog, data, M, N) for prog in population]
        
        new_pop = []
        
        # elitism
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        elite = [instr[:] for instr in population[best_idx]]  # deep copy
        new_pop.append(elite)
        
        while len(new_pop) < pop_size:
            # selection
            p1 = population[tournament_select(fitnesses, 0.75, 3)]
            p2 = population[tournament_select(fitnesses, 0.75, 3)]
            
            # crossover
            if random.random() < 0.9:
                c1, c2 = cross(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            
            # mutation
            if random.random() < 0.2:
                idx = random.randrange(len(c1))
                c1[idx] = mutate(c1[idx][:], M, N)
            if random.random() < 0.2:
                idx = random.randrange(len(c2))
                c2[idx] = mutate(c2[idx][:], M, N)
            
            new_pop.extend([c1, c2])
        
        population = new_pop[:pop_size]
        
        # progress
        best_fit = max(fitnesses)
        print(f"Gen {g+1}/{gens}  Best fitness: {best_fit}")
    
    # return best program
    fitnesses = [fitness(prog, data, M, N) for prog in population]
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    return population[best_idx], fitnesses[best_idx]

if __name__ == "__main__":
    best_prog, best_fit = ga(pop_size=20, gens=50, prog_len=10, M=100, N=100)
    print("\nBest program:", best_prog)
    print("Best fitness:", best_fit)