import math
import numpy as np
import random
from run_encoding_decoding_test import encode_network, decode_chromosome
from slopes import get_slope_angle
import matplotlib.pyplot as plt



START_VELOCITY = 20
MIN_VELOCITY = 1
MAX_VELOCITY = 25
INITIAL_GEAR = 7
INITIAL_BRAKE_TEMP = 500
AMBIENT_TEMP = 283
MAX_BRAKE_TEMP = 750
TAU = 30
HEAT_CONSTANT = 40
MAX_SLOPE_ANGLE = 10
DELTA_T = 0.1
DELTA_T_INT = 1
INITIAL_TIME_COUNTER = 20
GEAR_TIME_CONSTRAINT_INT = 20
SLOPE_LENGTH = 1000
PRESSURE_THRESHOLD = 0.01


def sigmoid(x, c=2):
    return 1 / (1 + np.exp(-c * x))


def tournament_select(fitness_list, tournament_selection_parameter, tournament_size):
    population_size = len(fitness_list)
    random_indices = random.sample(range(population_size), tournament_size)
    random_indices.sort(key=lambda idx: fitness_list[idx], reverse=True)
    for i in range(tournament_size):
        r = random.random()
        if r < tournament_selection_parameter:
            return random_indices[i]
    return random_indices[-1]


def cross(chromosome1, chromosome2):
    number_of_genes = len(chromosome1)
    cross_point = random.randint(1, number_of_genes - 1)
    new_chromosome_1 = np.concatenate((chromosome1[:cross_point], chromosome2[cross_point:]))
    new_chromosome_2 = np.concatenate((chromosome2[:cross_point], chromosome1[cross_point:]))
    return [new_chromosome_1, new_chromosome_2]


def mutate(chromosome, mutation_prob, creep_prob=0.8, creep_rate=0.5):
    mutated = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random() < mutation_prob:
            if random.random() < creep_prob:
                mutated[i] = mutated[i] - creep_rate / 2 + creep_rate * random.random()
                mutated[i] = max(0, min(1, mutated[i]))
            else:
                mutated[i] = 1 - mutated[i]
    return mutated


def truck_model(pressure, temp_b, temp_max, alpha_deg, gear, dt, current_velocity):
    CB = 3000.0
    MASS = 20000.0
    G = 9.82

    gear_forces = {
        1: 7.0, 2: 5.0, 3: 4.0, 4: 3.0, 5: 2.5,
        6: 2.0, 7: 1.6, 8: 1.4, 9: 1.2, 10: 1.0
    }
    feb = gear_forces[gear] * CB
    fg = MASS * G * math.sin(math.radians(alpha_deg))
    if temp_b < temp_max - 100:
        fb = MASS * G * pressure / 20.0
    else:
        fb = MASS * G * pressure * math.exp(-(temp_b - (temp_max - 100)) / 100.0) / 20.0
    total_force = fg - feb - fb
    acceleration = total_force / MASS
    new_velocity = current_velocity + acceleration * dt
    return new_velocity


def initialize_population(population_size, chromosome_length):
    return np.random.rand(population_size, chromosome_length)


def feedforward_neural_network(alpha_input, velocity_input, temp_input, wIH, wHO):
    input_neurons = np.array([alpha_input, velocity_input, temp_input])
    n_input_neurons = input_neurons.size
    n_hidden_neurons = wIH.shape[0]
    n_output_neurons = wHO.shape[0]
    hidden_neurons = np.zeros(n_hidden_neurons)
    output_neurons = np.zeros(n_output_neurons)
    for j in range(n_hidden_neurons):
        weights = wIH[j, :n_input_neurons]
        bias = wIH[j, -1]
        local_field = np.dot(weights, input_neurons) - bias
        hidden_neurons[j] = sigmoid(local_field)
    for i in range(n_output_neurons):
        weights = wHO[i, :n_hidden_neurons]
        bias = wHO[i, -1]
        local_field = np.dot(weights, hidden_neurons) - bias
        output_neurons[i] = sigmoid(local_field)
    pressure = output_neurons[0]
    delta_gear_raw = output_neurons[1]
    if delta_gear_raw > 0.6:
        delta_gear = 1
    elif delta_gear_raw < 0.4:
        delta_gear = -1
    else:
        delta_gear = 0
    return pressure, delta_gear

def Plot(horizontal_distance_list, alpha_list, pressure_list, gear_list, velocity_list, tempB_list, horizontal_distance_gear_list):
    plt.figure(figsize=(10, 12))
    plt.subplot(5, 1, 1)
    plt.plot(horizontal_distance_list, alpha_list, linewidth=1.5)
    plt.ylabel(r'$\alpha$ (deg)')
    plt.subplot(5, 1, 2)
    plt.plot(horizontal_distance_list, pressure_list, linewidth=1.5)
    plt.ylabel('Pressure')
    plt.subplot(5, 1, 3)
    plt.plot(horizontal_distance_gear_list, gear_list, linewidth=1.5)
    plt.ylabel('Gear')
    plt.subplot(5, 1, 4)
    plt.plot(horizontal_distance_list, velocity_list, linewidth=1.5)
    plt.ylabel('v (m/s)')
    plt.subplot(5, 1, 5)
    plt.plot(horizontal_distance_list, tempB_list, linewidth=1.5)
    plt.ylabel(r'$T_b$ (K)')
    plt.xlabel('Horizontal distance (m)')
    plt.tight_layout()
    plt.show()


def evaluate_individual(wIH, wHO, i_slope, i_dataset, plot_result=False):
    horizontal_distance = 0.0
    velocity = START_VELOCITY
    gear = INITIAL_GEAR
    brake_temp = INITIAL_BRAKE_TEMP
    time_counter = INITIAL_TIME_COUNTER
    valid = False

    alpha_list = []
    pressure_list = []
    gear_list = []
    velocity_list = []
    brake_temp_list = []
    horizontal_distance_list = []
    horizontal_distance_gear_list = []

    while not valid:
        alpha = get_slope_angle(horizontal_distance, i_slope, i_dataset)

        alpha_input = alpha / MAX_SLOPE_ANGLE
        velocity_input = velocity / MAX_VELOCITY
        temp_input = brake_temp / MAX_BRAKE_TEMP

        pressure, delta_gear = feedforward_neural_network(
            alpha_input, velocity_input, temp_input, wIH, wHO
        )

        alpha_list.append(alpha)
        pressure_list.append(pressure)
        gear_list.append(gear)
        velocity_list.append(velocity)
        brake_temp_list.append(brake_temp)
        horizontal_distance_list.append(horizontal_distance)
        horizontal_distance_gear_list.append(horizontal_distance)

        if time_counter % GEAR_TIME_CONSTRAINT_INT == 0:
            gear += delta_gear
            gear = min(max(gear, 1), 10)
            time_counter = 0
            gear_list.append(gear)
            horizontal_distance_gear_list.append(horizontal_distance)

        time_counter += DELTA_T_INT

        velocity = truck_model(
            pressure, brake_temp, MAX_BRAKE_TEMP, alpha, gear, DELTA_T, velocity
        )

        delta_temp_b = brake_temp - AMBIENT_TEMP
        if pressure < PRESSURE_THRESHOLD:
            d_delta_temp_b = -delta_temp_b / TAU
        else:
            d_delta_temp_b = HEAT_CONSTANT * pressure
        brake_temp = delta_temp_b + d_delta_temp_b * DELTA_T + AMBIENT_TEMP

        if brake_temp > MAX_BRAKE_TEMP:
            valid = True
        elif velocity < MIN_VELOCITY or velocity > MAX_VELOCITY:
            valid = True
        elif horizontal_distance >= SLOPE_LENGTH:
            valid = True
        else:
            horizontal_distance += np.cos(np.radians(alpha)) * velocity * DELTA_T

    avg_velocity = np.mean(velocity_list)
    fitness = avg_velocity * horizontal_distance

    if plot_result:
        Plot(horizontal_distance_list, alpha_list, pressure_list, gear_list, velocity_list, brake_temp_list, horizontal_distance_gear_list)

    return fitness

def genetic_algorithm():
    population = initialize_population(nFFNNs, chromosomeLength)
    creep_rate = creepRateInitial
    validation_counter = 0
    trained = False
    iterations = 0

    best_fitness_training_list = []
    best_fitness_validation_list = []
    best_validation_fitness_ever = -np.inf
    best_chromosome_so_far = None

    while not trained:
        iterations += 1
        max_fitness_training = -np.inf
        max_fitness_validation = -np.inf

        fitness_list_training = np.zeros(nFFNNs)
        fitness_list_validation = np.zeros(nFFNNs)

        for individual in range(nFFNNs):
            chromosome = population[individual]
            wIH, wHO = decode_chromosome(chromosome, nIn, nHidden, nOut, wMax)

            total_fitness = 0
            for iSlope in range(1, nTrainingSlopes + 1):
                total_fitness += evaluate_individual(wIH, wHO, iSlope, trainingSet, False)
            fitness_list_training[individual] = total_fitness / nTrainingSlopes

            if fitness_list_training[individual] > max_fitness_training:
                max_fitness_training = fitness_list_training[individual]
                best_chromosome_so_far = chromosome.copy()

            total_fitness_val = 0
            for iSlope in range(1, nValidationSlopes + 1):
                total_fitness_val += evaluate_individual(wIH, wHO, iSlope, validationSet, False)
            fitness_list_validation[individual] = total_fitness_val / nValidationSlopes

            if fitness_list_validation[individual] > max_fitness_validation:
                max_fitness_validation = fitness_list_validation[individual]

                if fitness_list_validation[individual] > best_validation_fitness_ever:
                    best_validation_fitness_ever = fitness_list_validation[individual]

        best_fitness_training_list.append(max_fitness_training)
        best_fitness_validation_list.append(max_fitness_validation)

        current_max_validation = max_fitness_validation
        if current_max_validation > best_validation_fitness_ever:
            best_validation_fitness_ever = current_max_validation
            validation_counter = 0
        else:
            validation_counter += 1

        if validation_counter >= trainedCriteria:
            trained = True

        if not trained:
            tmp_population = population.copy()
            for i in range(0, nFFNNs, 2):
                winner1 = tournament_select(fitness_list_training, tournamentProbability, tournamentSize)
                winner2 = tournament_select(fitness_list_training, tournamentProbability, tournamentSize)

                if random.random() < crossoverProb:
                    offspring1, offspring2 = cross(population[winner1], population[winner2])
                    tmp_population[i] = offspring1
                    if i + 1 < nFFNNs:
                        tmp_population[i + 1] = offspring2
                else:
                    tmp_population[i] = population[winner1]
                    if i + 1 < nFFNNs:
                        tmp_population[i + 1] = population[winner2]

            best_individual_index = np.argmax(fitness_list_training)
            tmp_population[0] = population[best_individual_index]

            for i in range(1, nFFNNs):
                tmp_population[i] = mutate(tmp_population[i], mutationProb, creepProb, creep_rate)

            if creep_rate > 0.005:
                creep_rate = creep_rate*creepDecay

            population = tmp_population

    plt.plot(range(iterations), best_fitness_training_list, ".-", label="Fitness training set")
    plt.plot(range(iterations), best_fitness_validation_list, ".-", label="Fitness validation set")
    plt.xlabel("Generation")
    plt.ylabel("Fitness value")
    plt.legend(loc='upper left')
    plt.show()

    print(f"iterations: {iterations}")
    return best_chromosome_so_far

nFFNNs = 10
nIn = 3
nHidden = 8
nOut = 2
wMax = 5
chromosomeLength = (nIn + 1) * nHidden + (nHidden + 1) * nOut
mutationProb = 1 / chromosomeLength
creepProb = 0.8
creepRateInitial = 0.5
creepDecay = 0.999
crossoverProb = 0.8
trainingSet = 1
validationSet = 2
nTrainingSlopes = 10
nValidationSlopes = 5
tournamentProbability = 0.75
tournamentSize = 4
trainedCriteria = 200

best_chromosome = genetic_algorithm()

with open("best_chromosome.py", "w") as f:
    f.write(" ".join(map(str, best_chromosome)))