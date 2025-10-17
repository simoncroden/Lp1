# reservoir_lorenz_predict.py
import numpy as np
import pandas as pd

# ---------------------------
# Parameters (assignment spec)
# ---------------------------
np.random.seed(42)            # reproducible results
n_u = 3                       # input dimension (x,y,z)
n_r = 500                     # reservoir size
k_ridge = 0.01                # ridge parameter
var_win = 0.002               # variance for input weights
var_wh = 2.0 / 500.0          # variance for reservoir weights
# ---------------------------

# Filenames (change if needed)
train_file = 'training-set.csv'   # or 'training-set.csv' depending on file
test_file = 'test-set-4.csv'        # or 'test-set.csv' depending on file

# Load data (assume no header)
X_train = pd.read_csv(train_file, header=None).to_numpy()  # shape (T_train, 3)
X_test  = pd.read_csv(test_file,  header=None).to_numpy()  # shape (T_test, 3)

T_train = X_train.shape[0]
T_test  = X_test.shape[0]

# Sanity
assert X_train.shape[1] == n_u, "Training data must have 3 columns"
assert X_test.shape[1]  == n_u, "Test data must have 3 columns"

# Initialize weights
W_in = np.random.normal(loc=0.0, scale=np.sqrt(var_win), size=(n_r, n_u))   # (n_r, 3)
W_h  = np.random.normal(loc=0.0, scale=np.sqrt(var_wh),  size=(n_r, n_r))   # (n_r, n_r)

# Initial reservoir state
r = np.zeros((n_r, 1))

# ---------------------------
# Training: collect reservoir states
# ---------------------------
# We'll collect r(t+1) after seeing x(t) and set target y(t+1) = x(t+1)
M = T_train - 1   # number of training state-target pairs
R = np.zeros((n_r, M))    # columns are reservoir states r(t+1)
Y = np.zeros((n_u, M))    # columns are targets x(t+1)

for t in range(M):
    u_t = X_train[t].reshape(n_u, 1)           # (3,1) = x(t)
    r = np.tanh(W_h @ r + W_in @ u_t)          # r(t+1)
    R[:, t] = r[:, 0]                          # store as column
    Y[:, t] = X_train[t + 1].reshape(n_u)      # target x(t+1)

# Ridge regression: W_out shape (n_u, n_r)
# Formula: W_out = Y R^T (R R^T + k I)^(-1)
RRt = R @ R.T                                # (n_r, n_r)
reg = k_ridge * np.eye(n_r)
inv_term = np.linalg.inv(RRt + reg)
W_out = (Y @ R.T) @ inv_term                 # (n_u, n_r)

# ---------------------------
# Feed test set (teacher forcing)
# ---------------------------
# Reset reservoir (assignment doesn't demand a particular start, zero is typical)
r = np.zeros((n_r, 1))
O_current = np.zeros((n_u, 1))

# We'll compute r and O for each test time step sequentially.
for t in range(T_test):
    u_t = X_test[t].reshape(n_u, 1)           # use test input x(t)
    r = np.tanh(W_h @ r + W_in @ u_t)         # r(t+1) with teacher input
    O_current = W_out @ r                     # O(t+1)
    # continue; final loop iteration leaves r and O_current equal to r(T) and O(T)?? 
    # Careful with indexing: after last iteration, O_current == O(T)

# After the loop:
# - r is the reservoir state after processing last test input (r(T))
# - O_current is O(T) (since O(t+1) is computed each iteration, final is O(T))

# ---------------------------
# Autonomous prediction for 501 steps
# Save O_2 (y-component) from O(T+1) to O(T+501) -> 501 values
# ---------------------------
n_predict = 501
pred_y = np.zeros((n_predict,))  # store y-component at each predicted time step

for step in range(n_predict):
    # use previous O_current as input
    r = np.tanh(W_h @ r + W_in @ O_current)  # r(t+1) where input is O(t)
    O_current = W_out @ r                    # O(t+1)
    pred_y[step] = float(O_current[1, 0])    # y-component (index 1)

# Save prediction.csv as single column, no header, comma separated
pd.DataFrame(pred_y).to_csv('prediction.csv', index=False, header=False)

print("Saved prediction.csv with shape:", pred_y.shape)
