import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv('iris-data.csv', header=None).to_numpy()
y = pd.read_csv('iris-labels.csv', header=None).to_numpy().flatten()

X = X / X.max()

n_x, n_y = 40, 40
n_features = X.shape[1]
epochs = 10
eta_0 = 0.1
de = 0.01
sigma_0 = 10
ds = 0.05

w = np.random.uniform(0, 1, size=(n_x, n_y, n_features))
w_initial = w.copy()

coords = np.array([[i,j] for i in range(n_x) for j in range(n_y)]).reshape(n_x,n_y,2)

def eta(epoch):
    return eta_0 * np.exp(-de * epoch)

def sigma(epoch):
    return sigma_0 * np.exp(-ds * epoch)

def h(i,j,i0,j0,s):
    dist_sq = (i-i0)**2 + (j-j0)**2
    return np.exp(-dist_sq/(2*s**2))

for epoch in range(epochs):
    for x in X:
        diff = w - x
        dist = np.linalg.norm(diff, axis=2)
        i0,j0 = np.unravel_index(np.argmin(dist), dist.shape)
        for i in range(n_x):
            for j in range(n_y):
                w[i,j] += eta(epoch) * h(i,j,i0,j0,sigma(epoch)) * (x - w[i,j])

def find_bmu(x, w):
    dist = np.linalg.norm(w - x, axis=2)
    return np.unravel_index(np.argmin(dist), dist.shape)

def get_bmu_positions(X, w):
    return np.array([find_bmu(x, w) for x in X])

pos_initial = get_bmu_positions(X, w_initial)
pos_final = get_bmu_positions(X, w)

colors = ['r','g','b']
label_names = ['Setosa','Versicolor','Virginica']

fig, axes = plt.subplots(1,2,figsize=(14,6))

for class_idx, color in enumerate(colors):
    idx = np.where(y==class_idx)[0]
    x_jitter = pos_initial[idx,0]
    y_jitter = pos_initial[idx,1]
    axes[0].scatter(x_jitter, y_jitter, c=color, label=label_names[class_idx], alpha=0.6, s=30)
axes[0].set_title("Initial random weights")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].legend()
axes[0].invert_yaxis()

for class_idx, color in enumerate(colors):
    idx = np.where(y==class_idx)[0]
    axes[1].scatter(pos_final[idx,0], pos_final[idx,1], c=color, label=label_names[class_idx], alpha=0.7)
axes[1].set_title("After training")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].invert_yaxis()

fig.suptitle("Self-Organizing Map", fontsize=14)
plt.tight_layout()
plt.show()
