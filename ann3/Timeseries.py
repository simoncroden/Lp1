import numpy as np
import pandas as pd

# Load training and test sets
training = pd.read_csv('training-set.csv', header=None)
testing = pd.read_csv('test-set-4.csv', header=None)

# Parameters
n_u = 3        # input neurons
n_r = 500      # reservoir neurons
dt = 0.02
np.random.seed(42)

# Initialize reservoir state
r = np.random.randn(n_r,1) * 0.01  # small random initial state

# Input weights w_in ~ N(0,0.002)
w_in = np.random.randn(n_r, n_u) * np.sqrt(0.002)

# Reservoir weights w_h ~ N(0,2/500) with spectral radius adjustment
w_h = np.random.randn(n_r, n_r) * np.sqrt(2/n_r)
rho = max(abs(np.linalg.eigvals(w_h)))
w_h *= 0.9 / rho  # scale so spectral radius = 0.9

# Collect reservoir states for training
T_train = training.shape[0]
R = np.zeros((n_r, T_train-1))
Y = np.zeros((n_u, T_train-1))

for t in range(T_train-1):
    x_t = training.iloc[t,:3].to_numpy().reshape(n_u,1)
    r = np.tanh(w_h @ r + w_in @ x_t)
    R[:,t] = r[:,0]
    Y[:,t] = training.iloc[t+1,:3].to_numpy()

# Ridge regression to compute output weights
k_ridge = 0.01
w_out = Y @ R.T @ np.linalg.inv(R @ R.T + k_ridge*np.eye(n_r))

# Reset reservoir and condition with test set
r = np.zeros((n_r,1))
T_test = testing.shape[0]
for t in range(T_test):
    x_t = testing.iloc[t,:3].to_numpy().reshape(n_u,1)
    r = np.tanh(w_h @ r + w_in @ x_t)

# Initialize output from last test step
O_current = w_out @ r

# Autonomous prediction loop (500 steps)
n_predict = 500
pred_y = np.zeros(n_predict)
for step in range(n_predict):
    r = np.tanh(w_h @ r + w_in @ O_current)
    O_current = w_out @ r
    pred_y[step] = O_current[1,0]  # save y-component

# Save prediction to CSV
pd.DataFrame(pred_y).to_csv('prediction.csv', index=False, header=False)
print("Saved prediction.csv with 500 steps (~10 seconds at dt=0.02)")
