import random
import numpy as np
import matplotlib.pyplot as plt
from function_data import load_function_data

def initialize_population(population_size, min_length, max_length, ops, var_Regs, all_Regs):
    population = []
    for _ in range(population_size):
        L = random.randrange(min_length, max_length+1)
        L += (4 - L%4)%4
        chrom = [random.randint(1,len(ops)) if i%4==0 else random.randint(1,len(var_Regs) if i%4==1 else len(all_Regs))
                 for i in range(L)]
        population.append({'Chromosome': chrom})
    return population

def evaluate_individual(chrom, varRegs, constRegs, maxLen, minLen,test=False):
    C_MAX = 1e7
    data = load_function_data()
    n=len(data)
    y_approx = np.zeros(n)
    y_true = np.zeros(n)
    x_points = np.zeros(n)

    for i,(x,y) in enumerate(data):
        regs = [x]+[0]*(len(varRegs)-1)+constRegs
        for j in range(0,len(chrom),4):
            op,dst,a,b = chrom[j:j+4]
            a,b = regs[a-1], regs[b-1]
            regs[dst-1] = [a+b, a-b, a*b, a/b if b!=0 else C_MAX][op-1]
        y_approx[i],y_true[i],x_points[i] = regs[0],y,x

    rmse = np.sqrt(np.mean((y_true-y_approx)**2))
    fitness = 1/rmse if rmse>0 else C_MAX

    if len(chrom)<minLen or len(chrom)>maxLen: 
        fitness/=C_MAX

    if test:
        return {
            'rootMeanSquareError': rmse,
            'xPoints': x_points,
            'yApprox': y_approx,
            'yTrue': y_true
        }
    
    return fitness

def tournament_select(fitness, prob, size):
    random_indexs = random.choices(range(len(fitness)), k=size)
    random_indexs.sort(key=lambda idx: fitness[idx], reverse=True)

    for i in range(size):
        r = random.random()
        if r < prob: 
            return random_indexs[i]
    
    return random_indexs[-1]

def TwoPointCrossover(w1, w2, pop, p):
    if random.random() < p:
        c1, c2 = pop[w1]['Chromosome'], pop[w2]['Chromosome']
        points = []
        for c in [c1, c2]:
            while True:
                r1, r2 = sorted(random.sample([i for i in range(1,len(c)+1) if i%4==0],2))
                points.append((r1,r2))
                break
        a1,b1 = points[0]
        a2,b2 = points[1]
        return [{'Chromosome': c1[:a1]+c2[a2:b2]+c1[b1:]}, {'Chromosome': c2[:a2]+c1[a1:b1]+c2[b2:]}]
    return [{'Chromosome': pop[w1]['Chromosome']}, {'Chromosome': pop[w2]['Chromosome']}]

def Mutate(chrom, ops, varRegs, allRegs, rate):
    nInst = len(chrom)//4
    prob = rate/len(chrom)
    genes = [chrom[i::4] for i in range(4)]
    for i in range(nInst):
        if random.random()<prob: 
            genes[0][i]=random.randint(1,len(ops))
        if random.random()<prob: 
            genes[1][i]=random.randint(1,len(varRegs))
        if random.random()<prob: 
            genes[2][i]=random.randint(1,len(allRegs))
        if random.random()<prob: 
            genes[3][i]=random.randint(1,len(allRegs))
    return {'Chromosome':[genes[i%4][i//4] for i in range(len(chrom))]}

def run_function_optimization(pop_size, min_len, max_len, ops, var_regs, all_regs,
                              n_gen, consts, t_prob, t_size, c_prob, m_rate,mDecay):
    
    population = initialize_population(pop_size, min_len, max_len, ops, var_regs, all_regs)
    fitness_list = np.zeros(pop_size)
    best_fitness = [0, 0]  
    global_best_chrom = None

    for gen in range(1, n_gen + 1):
        for i in range(pop_size):
            f = evaluate_individual(population[i]['Chromosome'],
                                    var_regs.copy(), consts, max_len, min_len)
            fitness_list[i] = f
            if f > best_fitness[0]:
                best_fitness = [f, i]
                global_best_chrom = population[i]['Chromosome'].copy()

        new_population = []
        for _ in range(0, pop_size, 2):
            w1 = tournament_select(fitness_list, t_prob, t_size)
            w2 = tournament_select(fitness_list, t_prob, t_size)
            children = TwoPointCrossover(w1, w2, population, c_prob)
            new_population.extend(children)

        elite_individual = {'Chromosome': global_best_chrom.copy()}
        new_population[0] = elite_individual

        for i in range(1, pop_size):
            chrom = new_population[i]['Chromosome']
            mutated = Mutate(chrom, ops, var_regs, all_regs, m_rate)
            new_population[i] = mutated

        m_rate *= mDecay
        population = new_population

        if gen % 1000 == 0 and global_best_chrom is not None:
            with open("best_chromosome.py", "w") as f:
                f.write(" ".join(map(str, global_best_chrom)))
            print(f"Gen {gen} RMS: {1 / best_fitness[0]:.6f}")

    if global_best_chrom is not None:
        with open("best_chromosome.py", "w") as f:
            f.write(" ".join(map(str, global_best_chrom)))
            print(f"Gen {gen} RMS: {1 / best_fitness[0]:.6f}")

nGen,popSize = 30,100
maxLen,minLen=150,25
tProb,tSize,cProb,mRate,mDecay = 0.75,5,0.8,80,0.9999
nVar=3
ops=[1,2,3,4]
consts=[1,2,3,4]
varRegs=[0]*nVar
allRegs=varRegs+consts

run_function_optimization(popSize,minLen,maxLen,ops,varRegs,allRegs, nGen,consts,tProb,tSize,cProb,mRate,mDecay)