import numpy as np
import matplotlib.pyplot as plt

# Sigmoid and sampling functions
sigmoid = lambda x: 1 / (1 + np.exp(-x))
def samp_pm(p): 
    return 2*(np.random.rand(*p.shape) < p).astype(int) - 1

# RBM training
def train_rbm(data, nv, nh, lr=0.05, epochs=2000, k=5):
    W = np.random.randn(nv, nh)
    b = np.zeros(nv)
    c = np.zeros(nh)
    for _ in range(epochs):
        ph = sigmoid(2*(data @ W + c))
        h = samp_pm(ph)
        v = data.copy()
        for _ in range(k):
            v = samp_pm(sigmoid(2*(h @ W.T + b)))
            h = samp_pm(sigmoid(2*(v @ W + c)))
        pos = data.T @ ph / len(data)
        neg = v.T @ sigmoid(2*(v @ W + c)) / len(v)
        W += lr * (pos - neg)
        b += lr * np.mean(data - v, axis=0)
        c += lr * np.mean(ph - sigmoid(2*(v @ W + c)), axis=0)
    return W, b, c

# RBM sampling
def sample_rbm(W, b, c, n=5000, burn=1000):
    v = samp_pm(np.full((1, W.shape[0]), 0.5))  # shape (1, nv)
    counts = {}
    for t in range(burn + n):
        h = samp_pm(sigmoid(2*(v @ W + c)))     # shape (1, nh)
        v = samp_pm(sigmoid(2*(h @ W.T + b)))   # shape (1, nv)
        if t >= burn:
            counts[tuple(v[0])] = counts.get(tuple(v[0]), 0) + 1
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

# KL divergence
def KL(p, q):
    return sum(p[x] * np.log(p[x] / (q.get(x, 1e-12))) for x in p)

# XOR dataset
data = np.array([[1,-1,-1],[1,1,1],[-1,-1,1],[-1,1,-1]])
p = {tuple(d):0.25 for d in data}

# Hidden units to test
Ms = [1,2,4,8]
kl = [np.inf]*len(Ms)

kl_best = [np.inf] * len(Ms)  # best KL per hidden unit size

# Train RBM multiple times
for it in range(25):
    for M_idx, M in enumerate(Ms):
        W, b, c = train_rbm(data, nv=3, nh=M, lr=0.1, epochs=2000, k=100)
        q = sample_rbm(W, b, c, n=5000)
        current_kl = KL(p, q)
        # Update best KL for this M if better
        if current_kl < kl_best[M_idx]:
            kl_best[M_idx] = current_kl
            best_W, best_b, best_c = W, b, c
    print(f"Iteration {it+1}: kl_best = {kl_best}")

print("Final best KL per hidden unit size:", kl_best)

# Optional theoretical curve for comparison
def Dkl(x):
    return np.log(2)*np.where(x<3, 3 - np.log(x+1)/np.log(2) - (x+1)/2**(np.log(x+1)/np.log(2)), 0)

x = np.array([1,2,4,8])
y = Dkl(x)

# Plot
plt.figure(figsize=(8,5))
plt.plot(Ms, kl_best, '-o', label='RBM KL')
plt.plot(x, y, '--', label='Theoretical Dkl')
plt.xlabel("Hidden units")
plt.ylabel("KL(p||q)")
plt.title("RBM on XOR")
plt.grid(True)
plt.legend()
plt.show()
