import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
training = pd.read_csv('iris-data.csv', header=None).to_numpy()
labels  = pd.read_csv('iris-labels.csv', header=None).to_numpy().flatten()

# --- Parameters ---
eta_0 = 0.1
de = 0.01
ds = 0.05
sigma_0 = 10
epochs = 10

n_x, n_y = 10, 10   # SOM grid
n_neurons = n_x * n_y
n_features = training.shape[1]

# --- Initialize weights ---
w = np.random.uniform(0, 1, (n_neurons, n_features))
w_initial = w.copy()

# --- Neuron grid coordinates ---
coords = np.array([(i, j) for i in range(n_x) for j in range(n_y)])

# --- Helper functions ---
def sigma(epoch):
    return sigma_0 * np.exp(-ds * epoch)

def eta(epoch):
    return eta_0 * np.exp(-de * epoch)

def h(i, i_0, epoch):
    dist2 = np.sum((coords[i] - coords[i_0])**2)
    return np.exp(-dist2 / (2 * sigma(epoch)**2))

def find_bmu(x, w):
    dists = np.linalg.norm(w - x, axis=1)
    return np.argmin(dists)

# --- Training ---
for epoch in range(epochs):
    for x in training:
        i_0 = find_bmu(x, w)
        for i in range(n_neurons):
            w[i] += eta(epoch) * h(i, i_0, epoch) * (x - w[i])

# --- BMU mapping ---
def get_bmu_positions(w, data):
    positions = []
    for x in data:
        i0 = find_bmu(x, w)
        positions.append(coords[i0])
    return np.array(positions)

pos_initial = get_bmu_positions(w_initial, training)
pos_final = get_bmu_positions(w, training)

# --- Plot results ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
colors = ['r', 'g', 'b']
label_names = ['Setosa', 'Versicolor', 'Virginica']

for class_idx, color in enumerate(colors):
    idx = np.where(labels == class_idx)[0]
    axes[0].scatter(pos_initial[idx, 0], pos_initial[idx, 1],
                    c=color, label=label_names[class_idx], alpha=0.7)
axes[0].set_title("Initial weight configuration")
axes[0].set_xlabel("SOM X")
axes[0].set_ylabel("SOM Y")
axes[0].legend()

for class_idx, color in enumerate(colors):
    idx = np.where(labels == class_idx)[0]
    axes[1].scatter(pos_final[idx, 0], pos_final[idx, 1],
                    c=color, label=label_names[class_idx], alpha=0.7)
axes[1].set_title(f"After training ({epochs} epochs)")
axes[1].set_xlabel("SOM X")
axes[1].set_ylabel("SOM Y")

plt.suptitle("Self-Organizing Map clustering of Iris data\n"
             "Red: Setosa, Green: Versicolor, Blue: Virginica", fontsize=12)
plt.tight_layout()
plt.show()
