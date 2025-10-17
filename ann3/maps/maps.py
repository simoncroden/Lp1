import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt 

training = pd.read_csv('iris-data.csv', header=None)
labels  = pd.read_csv('iris-labels.csv', header=None).to_numpy().flatten()

eta_0 = 0.1
de = 0.01
ds = 0.05
w = np.random.uniform(0,1,(40, 40))
x = []
delta_w = np.random.uniform(0,1,(40, 40))
sigma_0 = 10
r = np.arange(40)
epochs = 1

def sigma(epoch):
    return sigma_0 * np.exp(-ds*epoch)

def eta(epoch):
    return eta_0 * np.exp(-de*epoch)

def h(i,i_0,r,epoch):
    return np.exp(1/(2*sigma(epoch)**2)*abs(r[i]-r[i_0])**2)


for epoch in range(epochs):
    for x in training:
        # Find winning neuron (closest weight vector)
        dists = np.linalg.norm(w - x, axis=1)
        i_0 = np.argmin(dists)
        
        # Update all neurons
        for i in range(40):
            w[i] += eta(epoch) * h(i, i_0,r, epoch) * (x - w[i])



n_x, n_y = 10, 10   
coords = np.array([(i, j) for i in range(n_x) for j in range(n_y)])
w_initial = w.copy()

def find_bmu(x, w):
    dists = np.linalg.norm(w - x, axis=1)
    return np.argmin(dists)

def get_bmu_positions(w, data):
    positions = []
    for x in data:
        i0 = find_bmu(x, w)
        positions.append(coords[i0])
    return np.array(positions)


pos_initial = get_bmu_positions(w_initial, training)
pos_final = get_bmu_positions(w, training)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

colors = ['r', 'g', 'b']  # setosa, versicolor, virginica
label_names = ['Setosa', 'Versicolor', 'Virginica']

# Initial SOM mapping
for class_idx, color in enumerate(colors):
    idx = np.where(labels == class_idx)[0]
    axes[0].scatter(pos_initial[idx, 0], pos_initial[idx, 1],
                    c=color, label=label_names[class_idx], alpha=0.7)
axes[0].set_title("Initial weight configuration")
axes[0].set_xlabel("SOM X")
axes[0].set_ylabel("SOM Y")
axes[0].legend()

# Final SOM mapping
for class_idx, color in enumerate(colors):
    idx = np.where(labels == class_idx)[0]
    axes[1].scatter(pos_final[idx, 0], pos_final[idx, 1],
                    c=color, label=label_names[class_idx], alpha=0.7)
axes[1].set_title("After training (50 epochs)")
axes[1].set_xlabel("SOM X")
axes[1].set_ylabel("SOM Y")

plt.suptitle("Self-Organizing Map clustering of Iris data\n"
             "Red: Setosa, Green: Versicolor, Blue: Virginica", fontsize=12)
plt.tight_layout()
plt.show()