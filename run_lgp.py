import random
import numpy as np
import matplotlib.pyplot as plt
from function_data import load_function_data


def initialize_population(population_size, min_length, max_length, var_regs, all_regs):
    population = []

    for _ in range(population_size):
        length = random.randrange(min_length, max_length + 1)
        length += (4 - length % 4) % 4
        chromosome = []

        for i in range(length):
            if i % 4 == 0:
                gene = random.randint(1, 4)
            elif i % 4 == 1:
                gene = random.randint(1, len(var_regs))
            else:
                gene = random.randint(1, len(all_regs))
            chromosome.append(gene)

        population.append({'Chromosome': chromosome})

    return population


def evaluate_individual(chromosome, var_regs, const_regs, max_len, min_len):
    C_MAX = 1e8
    data = load_function_data()
    y_true = []
    y_approx = []

    for x, y in data:

        regs = [x] + [0] * (len(var_regs) - 1) + const_regs
        for i in range(0, len(chromosome), 4):
            op, dst, a, b = chromosome[i:i + 4]
            a_val, b_val = regs[a - 1], regs[b - 1]

            if op == 1:
                result = a_val + b_val
            elif op == 2:
                result = a_val - b_val
            elif op == 3:
                result = a_val * b_val
            else:
                result = a_val / b_val if b_val != 0 else C_MAX

            regs[dst - 1] = result

        y_true.append(y)
        y_approx.append(regs[0])

    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_approx)) ** 2))

    if rmse > 0:
        fitness = 1 / rmse 
    else:
        C_MAX   

    if not (min_len <= len(chromosome) <= max_len):
        fitness /= C_MAX

    return fitness

# DONE
def tournament_select(fitness_list, prob, size):
    random_indexs = random.choices(range(len(fitness_list)), k=size)
    random_indexs.sort(key=lambda idx: fitness_list[idx], reverse=True)

    for i in range(size):
        r = random.random()
        if r < prob:
            return random_indexs[i]
        
    return random_indexs[-1]


def cross(w1, w2, population, prob):
    if random.random() < prob:
        c1, c2 = population[w1]['Chromosome'], population[w2]['Chromosome']
        points = []
        for c in [c1, c2]:
            while True:
                r1, r2 = sorted(random.sample(
                    [i for i in range(1, len(c) + 1) if i % 4 == 0], 2))
                points.append((r1, r2))
                break
        
        a1, b1 = points[0]
        a2, b2 = points[1]

        return [
            {'Chromosome': c1[:a1] + c2[a2:b2] + c1[b1:]},
            {'Chromosome': c2[:a2] + c1[a1:b1] + c2[b2:]}
        ]
    
    return [
        {'Chromosome': population[w1]['Chromosome']},
        {'Chromosome': population[w2]['Chromosome']}
    ]


def mutate(chromosome, var_regs, all_regs, rate):
    n_inst = len(chromosome) // 4
    prob = rate / len(chromosome)
    genes = [chromosome[i::4] for i in range(4)]

    for i in range(n_inst):
        if random.random() < prob:
            genes[0][i] = random.randint(1, 4)
        if random.random() < prob:
            genes[1][i] = random.randint(1, len(var_regs))
        if random.random() < prob:
            genes[2][i] = random.randint(1, len(all_regs))
        if random.random() < prob:
            genes[3][i] = random.randint(1, len(all_regs))

    return {'Chromosome': [genes[i % 4][i // 4] for i in range(len(chromosome))]}


def run_function_optimization(pop_size, min_len, max_len, var_regs, all_regs,
                              n_gen, consts, t_prob, t_size, c_prob, m_rate, m_decay):

    population = initialize_population(pop_size, min_len, max_len, var_regs, all_regs)
    fitness_list = np.zeros(pop_size)
    best_fitness = [0, 0]
    global_best_chrom = None

    for gen in range(1, n_gen + 1):
        for i in range(pop_size):
            individual = evaluate_individual(
                population[i]['Chromosome'],
                var_regs.copy(), consts, max_len, min_len)
            fitness_list[i] = individual
            if individual > best_fitness[0]:
                best_fitness = [individual, i]
                global_best_chrom = population[i]['Chromosome'].copy()

        new_population = []
        for _ in range(0, pop_size, 2):
            w1 = tournament_select(fitness_list, t_prob, t_size)
            w2 = tournament_select(fitness_list, t_prob, t_size)
            children = cross(w1, w2, population, c_prob)
            new_population.extend(children)

        elite_individual = {'Chromosome': global_best_chrom.copy()}
        new_population[0] = elite_individual

        for i in range(1, pop_size):
            chrom = new_population[i]['Chromosome']
            mutated = mutate(chrom, var_regs, all_regs, m_rate)
            new_population[i] = mutated

        m_rate = m_rate*m_decay
        population = new_population

        if gen % 1000 == 0 and global_best_chrom is not None:
            with open("best_chromosome.py", "w") as f:
                f.write(" ".join(map(str, global_best_chrom)))
            print(f"Gen {gen} RMS: {1 / best_fitness[0]:.6f}")

    if global_best_chrom is not None:
        with open("best_chromosome.py", "w") as f:
            f.write(" ".join(map(str, global_best_chrom)))
        print(f"Final RMS: {1 / best_fitness[0]:.6f}")

# --- Parameters ---
n_gen, pop_size = 30, 100
max_len, min_len = 150, 25
t_prob, t_size, c_prob, m_rate, m_decay = 0.75, 5, 0.8, 80, 0.9999
n_var = 3
consts = [t for t in range(1,4)]
var_regs = [0] * n_var
all_regs = var_regs + consts

run_function_optimization(pop_size, min_len, max_len, var_regs, all_regs,
                          n_gen, consts, t_prob, t_size, c_prob, m_rate, m_decay)
