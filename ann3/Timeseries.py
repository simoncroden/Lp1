import numpy as np
import pandas as pd

training = pd.read_csv('training-set.csv', header=None)
testing  = pd.read_csv('test-set-4.csv', header=None)

n_u = 3 
n_r = 500 
dt = 0.02

r = np.random.randn(n_r,1) * 0.1

w_in = (np.random.randn(n_r, n_u)) * 0.05

w_h = np.random.randn(n_r, n_r) * np.sqrt(2/n_r)
rho = max(abs(np.linalg.eigvals(w_h)))
w_h *= 0.95 / rho


T_train = training.shape[1]
washout = 50
R = np.zeros((n_r, T_train-washout-1))
Y = np.zeros((n_u, T_train-washout-1))

for t in range(T_train-1):
    x_t = training.iloc[:, t].to_numpy().reshape(n_u,1)
    r = np.tanh(w_h @ r + w_in @ x_t)
    if t >= washout:
        R[:, t-washout] = r[:,0]
        Y[:, t-washout] = training.iloc[:, t+1].to_numpy()

k_ridge = 0.01
w_out = Y @ R.T @ np.linalg.inv(R @ R.T + k_ridge*np.eye(n_r))

r = np.random.randn(n_r,1)*0.1
T_test = testing.shape[1]
washout_test = 50
for t in range(T_test):
    x_t = testing.iloc[:, t].to_numpy().reshape(n_u,1)
    r = np.tanh(w_h @ r + w_in @ x_t)

O_current = w_out @ r
n_predict = 500
pred_y = np.zeros(n_predict)

for step in range(n_predict):
    r = np.tanh(w_h @ r + w_in @ O_current)
    O_current = w_out @ r
    pred_y[step] = O_current[1,0]  

pd.DataFrame(pred_y).to_csv('prediction.csv', index=False, header=False)
print("Saved prediction.csv with 500 y-component predictions (~10 s)")
