import numpy as np
from run_encoding_decoding_test import decode_chromosome
from run_ffnn_optimization import evaluate_individual

n_input = 3
n_hidden = 4
n_output = 2
w_max = 5
test_set = 3
n_test_slopes = 5

i_data_set = test_set
i_slope = 2

with open("best_chromosome.py", "r") as f:
    chromosome = [float(x) for x in f.read().split()]

w_input_hidden, w_hidden_output = decode_chromosome(chromosome, n_input, n_hidden, n_output, w_max)

plot_result = True
fitness = evaluate_individual(w_input_hidden, w_hidden_output, i_slope, i_data_set, plot_result)

print(f"Fitness = {fitness:.6f}")
