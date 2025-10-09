import math
from genetic_algorithm import run_function_optimization

population_size = 100               # Do NOT change
maximum_variable_value = 5          # Do NOT change: (x_i in [-a,a], where a = maximumVariableValue)
number_of_genes = 50                # Do NOT change
number_of_variables = 2  	    # Do NOT change

tournament_size = 4                 # Changes allowed
tournament_probability = 0.65       # Changes allowed
crossover_probability = 0.7         # Changes allowed
mutation_probability = 0.02         # Changes allowed. (Note: 0.02 <=> 1/numberOfGenes)
number_of_generations = 3000        # Changes allowed.

for i in range(10):
    [maximum_fitness, x_best] = run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, tournament_size, \
                                        tournament_probability, crossover_probability, mutation_probability, number_of_generations);
    g = (1.5-x_best[0]+x_best[0]*x_best[1])**2 + (2.25-x_best[0]+x_best[0]*x_best[1]**2)**2 + (2.625-x_best[0]+x_best[0]*x_best[1]**3)**2
    output = f"{g:.20f}  {x_best[0]:.8f}  {x_best[1]:.8f}"
    print(output)
