import sympy as sp
import numpy as np
from function_data import load_function_data
import matplotlib.pyplot as plt

def evaluate_individual(chrom, varRegs, constRegs, maxLen, minLen,test=False):
    C_MAX = 1e8
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


def estimate_function(chromosome):
    const_registers = ["1", "3", "-1", "2"]
    variable_registers = ["x", "0", "0"]
    registers = variable_registers + const_registers

    for i in range(0, len(chromosome), 4):
        op_index, dst_index, reg_a_index, reg_b_index = chromosome[i:i + 4]
        reg_a = registers[reg_a_index - 1]
        reg_b = registers[reg_b_index - 1]

        operators = [
            f"({reg_a}+{reg_b})",
            f"({reg_a}-{reg_b})",
            f"({reg_a}*{reg_b})",
            f"({reg_a}/{reg_b})" if reg_b != "0" else "1e10"
        ]

        registers[dst_index - 1] = operators[op_index - 1]

    return sp.simplify(sp.sympify(registers[0]))




maxLen,minLen=150,25
nVar=3
consts=[1,2,3,4]
varRegs=[0]*nVar

# --- Final evaluation ---
with open("best_chromosome.py","r") as f: 
    chrom=[int(x) for x in f.read().split()]
evalRes = evaluate_individual(chrom,varRegs.copy(),consts,maxLen,minLen,test=True)
plt.figure(figsize=(8,5))
plt.plot(evalRes['xPoints'],evalRes['yTrue'],'-',linewidth=1.5,label="g(x)")
plt.plot(evalRes['xPoints'],evalRes['yApprox'],'--',linewidth=1.5,label="Approx")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
print(f"Estimated function: {estimate_function(chrom)}")
print(f"RMS error: {evalRes['rootMeanSquareError']*100:.2f}%")