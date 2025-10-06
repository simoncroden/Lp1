###########################
#
# Ant System (AS) for TSP
#
###########################

import math
import numpy as np
import matplotlib.pyplot as plt
import random

def initialize_pheromone_levels(number_of_cities, tau_0):
  pheromone_matrix = np.zeros((number_of_cities, number_of_cities))
  for i in range(number_of_cities):
    for j in range(number_of_cities):
      pheromone_matrix[i][j] = tau_0
  return pheromone_matrix

def get_visibility(city_locations):
  number_of_cities = len(city_locations)
  n = np.zeros((number_of_cities, number_of_cities))

  for i in range(number_of_cities):
    for j in range(number_of_cities):
      if i != j:
        n[i][j] = 1/(math.sqrt((city_locations[i][0] - city_locations[j][0])**2 + (city_locations[i][1] - city_locations[j][1])**2))

  return n

def generate_path(pheromone_levels, visibility, alpha, beta):
  n_nodes = len(pheromone_levels)
  tabu_list = [np.random.randint(n_nodes)]

  while len(tabu_list) < n_nodes:
    current_node = tabu_list[-1]

    possible_nodes = [i for i in range(n_nodes) if i not in tabu_list]

    num = np.array([
            (pheromone_levels[current_node][i] ** alpha) * (visibility[current_node][i] ** beta)
            for i in possible_nodes
        ])
    
    probs = num / num.sum()
    tabu_list.append(np.random.choice(possible_nodes, p=probs))


  tabu_list.append(tabu_list[0])
  return tabu_list

def get_path_length(path, city_locations):
  length = 0
  for i in range(1,len(path)):
    length += math.sqrt((city_locations[path[i]][0]-city_locations[path[i-1]][0])**2 + (city_locations[path[i]][1]-city_locations[path[i-1]][1])**2)
  return length

def compute_delta_pheromone_levels(path_collection, path_length_collection):

  n_nodes = max(max(path) for path in path_collection) + 1
  delta = np.zeros((n_nodes, n_nodes))
                   
  for k, path in enumerate(path_collection): 
    delta_k = 1 / path_length_collection[k]
    for i in range(len(path)-1):
      a, b = path[i], path[i+1]
      delta[a, b] += delta_k
  return delta

def update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho):
  pheromone_levels = (1-rho)*pheromone_levels + delta_pheromone_levels
  pheromone_levels[pheromone_levels < 1e-15] = 1e-15
  return pheromone_levels


##################################################
#  Plots the cities (nodes):
##################################################

def plot_path(city_locations, path):
    x = [city_locations[i][0] for i in path]
    y = [city_locations[i][1] for i in path]
    plt.figure()
    plt.plot(x, y, 'o-')
    plt.scatter(x[0], y[0])
    plt.title(f"length={get_path_length(path, city_locations):.2f}")
    plt.show()

#####################################
# Main program:
#####################################

###########################
# Data:
###########################
from city_data import city_locations
number_of_cities = len(city_locations)

###########################
# Parameters:
###########################
number_of_ants = 50 ## Changes allowed.
alpha = 1.0         ## Changes allowed.
beta = 5.0          ## Changes allowed.
rho = 0.5           ## Changes allowed.
tau_0 = 0.1         ## Changes allowed.

target_path_length = 99.9999999

#################################
# Initialization:
#################################

pheromone_levels = initialize_pheromone_levels(number_of_cities, tau_0)
visibility = get_visibility(city_locations)

#################################
# Main loop:
#################################

iteration_index = 0
minimum_path_length = math.inf
path_length = math.inf


while (minimum_path_length > target_path_length):
  iteration_index += 1
  path_collection = []
  path_length_collection = []
  for ant_index in range(number_of_ants):  
    path = generate_path(pheromone_levels, visibility, alpha, beta)
    path_length = get_path_length(path, city_locations)
    if (path_length < minimum_path_length):
      minimum_path_length = path_length
      print(minimum_path_length)
      print(path)
      plot_path(city_locations,path)

    path_collection.append(path)
    path_length_collection.append(path_length)
  delta_pheromone_levels = compute_delta_pheromone_levels(path_collection,path_length_collection)
  pheromone_levels = update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho)

input(f'Press return to exit')