import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from function_data import load_function_data

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

def TournamentSelect(fitness, prob, size):
    tour = random.choices(range(len(fitness)), k=size)
    sortedTour = sorted(tour, key=lambda i: -fitness[i])
    for i in sortedTour:
        if random.random()<prob: return i
    return sortedTour[0]

def InitializePopulation(popSize, minLen, maxLen, ops, varRegs, allRegs):
    pop=[]
    for _ in range(popSize):
        L = random.randrange(minLen, maxLen+1)
        L += (4 - L%4)%4
        nInst = L//4
        chrom = [random.randint(1,len(ops)) if i%4==0 else random.randint(1,len(varRegs) if i%4==1 else len(allRegs))
                 for i in range(L)]
        pop.append({'Chromosome': chrom})
    return pop

def EvaluateIndividual(chrom, ops, varRegs, constRegs, maxLen, minLen, test=False):
    cMax = 1e7
    data = load_function_data()
    n=len(data)
    yA,yT,xs = np.zeros(n), np.zeros(n), np.zeros(n)
    for i,(x,y) in enumerate(data):
        regs = [x]+[0]*(len(varRegs)-1)+constRegs
        for j in range(0,len(chrom),4):
            op,dst,a,b = chrom[j:j+4]
            a,b = regs[a-1], regs[b-1]
            regs[dst-1] = [a+b, a-b, a*b, a/b if b!=0 else cMax][op-1]
        yA[i],yT[i],xs[i] = regs[0],y,x
    rmse = np.sqrt(np.mean((yT-yA)**2))
    fitness = 1/rmse if rmse>0 else cMax
    if len(chrom)<minLen or len(chrom)>maxLen: 
        fitness/=cMax
    return {'rootMeanSquareError':rmse,'xPoints':xs,'yApprox':yA,'yTrue':yT} if test else fitness

def EstimateFunction(chrom, ops):
    constRegs = ["1","3","-1","2"]
    varRegs = ["x","0","0"]
    regs=varRegs+constRegs
    for i in range(0,len(chrom),4):
        op,dst,a,b = chrom[i:i+4]
        a,b = regs[a-1], regs[b-1]
        regs[dst-1] = [f'({a}+{b})',f'({a}-{b})',f'({a}*{b})',f'({a}/{b})' if b!="0" else "1e7"][op-1]
    return sp.simplify(sp.sympify(regs[0]))

nGen,popSize = 30,100
maxLen,minLen=150,25
tProb,tSize,cProb,mRate,mDecay = 0.75,5,0.8,80,0.9999
nVar=3
ops=[1,2,3,4]
consts=[1,3,-1,2]
varRegs=[0]*nVar
allRegs=varRegs+consts

pop = InitializePopulation(popSize,minLen,maxLen,ops,varRegs,allRegs)
fitness = np.zeros(popSize)
best=[0,0]

for gen in range(1,nGen+1):
    for i in range(popSize):
        f = EvaluateIndividual(pop[i]['Chromosome'],ops,varRegs.copy(),consts,maxLen,minLen)
        fitness[i]=f
        if f>best[0]: 
            best=[f,i]
    newPop=[]
    for _ in range(popSize//2):
        w1,w2 = TournamentSelect(fitness,tProb,tSize),TournamentSelect(fitness,tProb,tSize)
        newPop.extend(TwoPointCrossover(w1,w2,pop,cProb))
    newPop[0] = pop[best[1]]
    for i in range(1,popSize): 
        newPop[i]=Mutate(newPop[i]['Chromosome'],ops,varRegs,allRegs,mRate)
    mRate*=mDecay
    pop=newPop
    if gen%1000==0:
        with open("BestChromosome.txt","w") as f: 
            f.write(" ".join(map(str,pop[0]['Chromosome'])))
        print(f"Gen {gen} RMS: {1/best[0]:.6f}")

with open("BestChromosome.txt","r") as f: 
    chrom=[int(x) for x in f.read().split()]
evalRes = EvaluateIndividual(chrom,ops,varRegs.copy(),consts,maxLen,minLen,test=True)
plt.figure(figsize=(8,5))
plt.plot(evalRes['xPoints'],evalRes['yTrue'],'-',linewidth=1.5,label="g(x)")
plt.plot(evalRes['xPoints'],evalRes['yApprox'],'--',linewidth=1.5,label="Approx")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
print(f"Estimated function: {EstimateFunction(chrom,ops)}")
print(f"RMS error: {evalRes['rootMeanSquareError']*100:.2f}%")
