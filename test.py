#!/usr/bin/env python3
"""
run_lgp.py
Linear Genetic Programming for inferring rational functions g(x) = P(x)/Q(x)
- Chromosomes: list of instructions, each instruction = [op, dst, src1, src2]
  op: 1:+, 2:-, 3:*, 4:/
  dst: destination variable register index (1..M)
  src1, src2: indices into combined (variable_registers + constant_registers), 1..M+N

- Uses tournament selection, two-point crossover (between instructions), mutation.
- Enforces hard limit: max_instructions = 100 (=> max genes = 400). Chromosomes exceeding
  this limit after crossover are discarded and selection/crossover retried.
- At the end it stores the best chromosome (and meta) in best_chromosome.py
"""

import random
import copy
import math
import json
from function_data import load_function_data
import numpy as np

# ---------- Parameters (tune as needed) ----------
M = 50              # number of variable registers
N = 50               # number of constant registers
POP_SIZE = 400       # population size (recommend >= 100)
PROG_LEN = 40        # initial number of instructions per program
MAX_INSTR = 100      # hard limit on instructions (per assignment)
GENERATIONS = 100000
TOURNAMENT_SIZE = 3
TOURNAMENT_PROB = 0.75
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.25  # probability PER FIELD per instruction (see mutate)
USE_UNIFORM_PROB = 0.25  # fraction of crossovers that use uniform crossover
STAGNATION_INJECT_THRESHOLD = 10
INJECT_COUNT = 5

# small epsilon to avoid division by zero
EPS = 1e-12

# ---------- Utilities ----------
def make_random_instruction(M, N):
    return [
        random.randint(1, 4),         # op
        random.randint(1, M),         # dst
        random.randint(1, M + N),     # src1
        random.randint(1, M + N)      # src2
    ]

def initialize_population(M, N, pop_size, prog_len):
    pop = []
    for _ in range(pop_size):
        prog = [make_random_instruction(M, N) for _ in range(prog_len)]
        pop.append(prog)
    return pop

def operators(a, b, op):
    if op == 1:
        return a + b
    if op == 2:
        return a - b
    if op == 3:
        return a * b
    if op == 4:
        # safe division
        return a / b if b != 0 else 0.0
    raise ValueError("Unknown op")

def decoder_step(instr, var_regs, const_regs):
    all_regs = var_regs + const_regs
    op1 = all_regs[instr[2] - 1]
    op2 = all_regs[instr[3] - 1]
    var_regs[instr[1] - 1] = operators(op1, op2, instr[0])
    return var_regs

def evaluate_individual(program, const_regs, data, M):
    """Return (fitness, rmse). fitness = 1 / rmse."""
    y_preds = []
    for (xk, yk) in data:
        var_regs = [0.0] * M
        var_regs[0] = float(xk)
        for instr in program:
            var_regs = decoder_step(instr, var_regs, const_regs)
        y_preds.append(var_regs[0])
    errors = [(yk - yhat) ** 2 for (_, yk), yhat in zip(data, y_preds)]
    mse = float(np.mean(errors)) if len(errors) > 0 else float('inf')
    rmse = math.sqrt(mse)
    fitness = 1.0 / (rmse + EPS)
    return fitness, rmse

# ---------- Selection ----------
def tournament_select(fitness_list, tournament_probability, tournament_size):
    pop_n = len(fitness_list)
    chosen = random.sample(range(pop_n), tournament_size)
    chosen.sort(key=lambda i: fitness_list[i], reverse=True)
    for idx in chosen:
        if random.random() < tournament_probability:
            return idx
    return chosen[-1]

# ---------- Crossover (instruction-level) ----------
def two_point_instruction_crossover(p1, p2):
    """
    Two-point crossover *between instructions*.
    p1, p2 are lists of instructions.
    Returns children (c1, c2) where each child is a list of instructions.
    """
    n = len(p1)
    if n <= 1:
        return copy.deepcopy(p1), copy.deepcopy(p2)
    i1, i2 = sorted(random.sample(range(1, n), 2))  # split points between instructions
    # ensure decent segment length
    min_len = max(1, n // 10)
    if i2 - i1 < min_len:
        i2 = min(i1 + min_len, n - 1)
    c1 = copy.deepcopy(p1[:i1] + p2[i1:i2] + p1[i2:])
    c2 = copy.deepcopy(p2[:i1] + p1[i1:i2] + p2[i2:])
    return c1, c2

def uniform_instruction_crossover(p1, p2):
    n = len(p1)
    c1, c2 = [], []
    for i in range(n):
        if random.random() < 0.5:
            c1.append(copy.deepcopy(p1[i])); c2.append(copy.deepcopy(p2[i]))
        else:
            c1.append(copy.deepcopy(p2[i])); c2.append(copy.deepcopy(p1[i]))
    return c1, c2

# ---------- Mutation ----------
def mutate(program, M, N, mutation_prob):
    """Per-field mutation: each of the 4 fields in each instruction mutates independently
       with probability mutation_prob."""
    out = []
    for instr in program:
        new_instr = instr[:]  # shallow copy list of 4 ints
        # mutate op
        if random.random() < mutation_prob:
            new_instr[0] = random.randint(1, 4)
        # mutate dst
        if random.random() < mutation_prob:
            new_instr[1] = random.randint(1, M)
        # mutate src1
        if random.random() < mutation_prob:
            new_instr[2] = random.randint(1, M + N)
        # mutate src2
        if random.random() < mutation_prob:
            new_instr[3] = random.randint(1, M + N)
        out.append(new_instr)
    return out

# ---------- Save best chromosome ----------
def save_best(best_chrom, M, N, const_regs, filename="best_chromosome.py"):
    """Write a small python file that defines BEST_CHROMOSOME, M, N, CONSTANT_REGISTERS."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by run_lgp.py\n")
        f.write("M = %d\n" % M)
        f.write("N = %d\n" % N)
        f.write("CONSTANT_REGISTERS = %s\n\n" % (repr(const_regs)))
        f.write("BEST_CHROMOSOME = [\n")
        for instr in best_chrom:
            f.write("    %s,\n" % repr(instr))
        f.write("]\n")
    print(f"Saved best chromosome to {filename}")

# ---------- Main GA ----------
def run_ga(
    M=M, N=N, pop_size=POP_SIZE, prog_len=PROG_LEN, max_instr=MAX_INSTR,
    gens=GENERATIONS, tournament_size=TOURNAMENT_SIZE, tournament_prob=TOURNAMENT_PROB,
    crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB, use_uniform_prob=USE_UNIFORM_PROB,
    stagnation_inject_threshold=STAGNATION_INJECT_THRESHOLD, inject_count=INJECT_COUNT
):
    data = load_function_data()
    # initialize population (programs are lists of instructions)
    population = initialize_population(M, N, pop_size, prog_len)
    # constants: set once and for all (random floats)
    constant_registers = [random.uniform(-10.0, 10.0) for _ in range(N)]

    best_chromosome = None
    best_fitness = -float("inf")
    stagnation_counter = 0

    for g in range(1, gens + 1):
        fitnesses = []
        rmses = []
        for prog in population:
            f, rmse = evaluate_individual(prog, constant_registers, data, M)
            fitnesses.append(f)
            rmses.append(rmse)

        # generation best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_f = fitnesses[gen_best_idx]
        gen_best_rmse = rmses[gen_best_idx]

        # update global best
        if gen_best_f > best_fitness + EPS:
            best_fitness = gen_best_f
            best_chromosome = copy.deepcopy(population[gen_best_idx])
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        print(f"Gen {g}/{gens}  gen-best-fit={gen_best_f:.6g}  gen-RMSE={gen_best_rmse:.6g}  global-best={best_fitness:.6g} stagn={stagnation_counter}")

        # adaptive mutation multiplier
        current_mutation = min(0.9, mutation_prob * (1 + stagnation_counter / 5.0))

        # produce new population ensuring child length <= max_instr
        new_pop = []
        # keep top-2 elites
        sorted_idxs = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        for k in range(2):
            if len(new_pop) < pop_size:
                new_pop.append(copy.deepcopy(population[sorted_idxs[k]]))

        attempts = 0
        while len(new_pop) < pop_size:
            attempts += 1
            if attempts > pop_size * 50:
                # emergency: fill remaining randomly to avoid infinite loops
                while len(new_pop) < pop_size:
                    new_pop.append(initialize_population(M, N, 1, prog_len)[0])
                break

            i1 = tournament_select(fitnesses, tournament_prob, tournament_size)
            i2 = tournament_select(fitnesses, tournament_prob, tournament_size)
            p1 = copy.deepcopy(population[i1])
            p2 = copy.deepcopy(population[i2])

            if random.random() < crossover_prob:
                if random.random() < use_uniform_prob:
                    c1, c2 = uniform_instruction_crossover(p1, p2)
                else:
                    c1, c2 = two_point_instruction_crossover(p1, p2)
            else:
                c1, c2 = p1, p2

            # enforce hard limit on instructions
            if len(c1) <= max_instr and len(c2) <= max_instr:
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)
            # else discard and retry

        # mutation
        population = [mutate(ind, M, N, current_mutation) for ind in new_pop]

        # inject random programs if stagnation persists
        if stagnation_counter >= stagnation_inject_threshold:
            for j in range(inject_count):
                idx = pop_size - 1 - j
                if idx >= 2:  # keep elites
                    population[idx] = initialize_population(M, N, 1, prog_len)[0]
            print(f"  >>> injected {inject_count} random programs due to stagnation")

    # save best
    save_best(best_chromosome, M, N, constant_registers)
    return best_fitness, best_chromosome

# ---------- If run as script ----------
if __name__ == "__main__":
    random.seed(42)
    best_f, best_prog = run_ga()
    print("Done. Best fitness:", best_f)
    print("Best program (first 10 instructions):")
    for instr in best_prog[:10]:
        print(instr)
