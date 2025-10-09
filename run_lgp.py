import numpy as np
import random
from function_data import load_function_data


def tournament_select(pop, fitnesses, k=3):
    # pick k random indices
    participants = random.sample(range(len(pop)), k)
    best = participants[0]
    best_f = fitnesses[best]
    for idx in participants[1:]:
        if fitnesses[idx] > best_f:  # assumes maximization
            best = idx
            best_f = fitnesses[idx]
    return best

def two_point_crossover(parent1, parent2):
    L = len(parent1)
    if L < 2:
        return parent1[:], parent2[:]
    pt1, pt2 = sorted(random.sample(range(1, L), 2))
    child1 = parent1[:]
    child2 = parent2[:]
    child1[pt1:pt2] = parent2[pt1:pt2]
    child2[pt1:pt2] = parent1[pt1:pt2]
    return child1, child2

def mutate(chrom, mutation_rate=0.1, mutation_std=0.1):
    # chrom must be a list (if NumPy array, convert to list or adapt)
    new_chrom = chrom[:]
    for i in range(len(new_chrom)):
        if random.random() < mutation_rate:
            new_chrom[i] += random.gauss(0.0, mutation_std)
    return new_chrom

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

M = 10
N = 10
variable_registers = [1] * M
constant_registers = [random.randint(1, 100) for _ in range(N)]

data = load_function_data()

#print(instructions)
#print(decoder(instructions[0],variable_registers,constant_registers))

def fitness(instructions, data, M=10, N=10):

    # run each instruction
    for instr in instructions:
        decoder(instr, variable_registers, constant_registers)
    
    # here: just return negative squared error to some target
    # (replace with your real objective)
    target_row = data[0]  # assume it's a list of length M
    fitness_val = -sum(abs(v - t) for v, t in zip(variable_registers, target_row))
    return fitness_val

def ga(pop_size=20, gens=50, prog_len=10, M=10, N=10):
    data = load_function_data()
    
    instructions = []

    for i in range(10):
        instruction = []
        instruction.append(random.randint(1,4))
        instruction.append(random.randint(1,M))
        instruction.append(random.randint(1,M+N))
        instruction.append(random.randint(1,M+N))
        instructions.append(instruction)
    
    # evolution loop
    for g in range(gens):
        fitnesses = [fitness(p, data, M, N) for p in instructions]
        
        new_pop = []
        
        # elitism: keep best
        best_idx = max(range(len(instructions)), key=lambda i: fitnesses[i])
        new_pop.append(instructions[best_idx])
        
        while len(new_pop) < pop_size:
            # selection
            p1 = instructions[tournament_select(instructions, fitnesses)]
            p2 = instructions[tournament_select(instructions, fitnesses)]
            
            # crossover
            if random.random() < 0.9:
                c1, c2 = two_point_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            
            # mutation (simple: mutate random instruction)
            if random.random() < 0.2:
                idx = random.randrange(len(c1))
                c1[idx] = [
                    random.randint(1,4),
                    random.randint(1,M),
                    random.randint(1,M+N),
                    random.randint(1,M+N)
                ]
            if random.random() < 0.2:
                idx = random.randrange(len(c2))
                c2[idx] = [
                    random.randint(1,4),
                    random.randint(1,M),
                    random.randint(1,M+N),
                    random.randint(1,M+N)
                ]
            
            new_pop.extend([c1, c2])
        
        instructions = new_pop[:pop_size]
        
        # progress
        best_fit = max(fitnesses)
        print(f"Gen {g+1}/{gens}  Best fitness: {best_fit}")
    
    # return best program
    fitnesses = [fitness(p, data, M, N) for p in instructions]
    best_idx = max(range(len(instructions)), key=lambda i: fitnesses[i])
    return instructions[best_idx], fitnesses[best_idx]

if __name__ == "__main__":
    best_prog, best_fit = ga()
    print("\nBest program:", best_prog)
    print("Best fitness:", best_fit)