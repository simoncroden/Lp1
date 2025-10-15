import sympy as sp
import numpy as np
from function_data import load_function_data
import matplotlib.pyplot as plt


def evaluate_individual(chromosome, var_regs, const_regs):
    C_MAX = 1e8
    data = load_function_data()
    y_true = []
    y_approximation = []
    x_points = np.zeros(len(data))

    for x, y in data:

        regs = [x] + [0] * (len(var_regs) - 1) + const_regs
        for i in range(0, len(chromosome), 4):
            operation, destination, register_1, register_2 = chromosome[i:i + 4]
            register_1_val, register_2_val = regs[register_1 - 1], regs[register_2 - 1]

            if operation == 1:
                result = register_1_val+register_2_val
            elif operation == 2:
                result = register_1_val-register_2_val
            elif operation == 3:
                result = register_1_val*register_2_val
            else:
                result = register_1_val/register_2_val if register_2_val != 0 else C_MAX

            regs[destination - 1] = result

        y_true.append(y)
        y_approximation.append(regs[0])
        x_points.append(x)

    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_approximation)) ** 2))

    return {
        'root_mean_square_error': rmse,
        'x_points': x_points,
        'y_approx': y_approximation,
        'y_true': y_true
    }

def estimate_function(chromosome):
    const_registers = ["1", "2", "3", "4"]
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

number_of_variable_registers = 3
constants = [1, 2, 3, 4]
var_regs = [0] * number_of_variable_registers

with open("best_chromosome.py", "r") as f:
    chrom = [int(x) for x in f.read().split()]

evaluate = evaluate_individual(chrom, var_regs.copy(), constants)

plt.figure(figsize=(8, 5))
plt.plot(evaluate['x_points'], evaluate['y_true'], '-', linewidth=1.5, label="g(x)")
plt.plot(evaluate['x_points'], evaluate['y_approx'], '--', linewidth=1.5, label="Approx")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"Estimated function: {estimate_function(chrom)}")
print(f"RMS error: {evaluate['root_mean_square_error'] * 100:.2f}%")
