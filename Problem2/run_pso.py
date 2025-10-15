import math
import numpy as np
import matplotlib.pyplot as plt
import random

def f(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def init(N,n,min,max):
    ALPHA = 1
    DELTA_T = 1
    x_matrix = np.zeros((N, n))
    velocity_matrix = np.zeros((N, n))
    for i in range(N):
        for j in range(n):
            r = np.random.rand()
            x_matrix[i][j] = min + r*(max-min)
            velocity_matrix[i][j] = ALPHA/DELTA_T*(-(max-min)/2+r*(max-min))
    return x_matrix,velocity_matrix

def algo(x_matrix,velocity_matrix,iterations,delta_t,v_max):
    N,n = x_matrix.shape
    C1 = 1.5
    C2 = 1.5    

    x_best = x_matrix.copy()
    x_best_values = np.array([f(x_matrix[i]) for i in range(N)])
    x_global_best = x_best[np.argmin(x_best_values)]

    for iteration in range(iterations):
        for i in range(N):
            w = 1.2 - 0.9*i/iterations
            random_1, random_2= np.random.rand(), np.random.rand()

            velocity_matrix[i] = (
               w*velocity_matrix[i] +C1*random_2*(x_best[i]-x_matrix[i])/delta_t + C2*random_1*(x_global_best-x_matrix[i])/delta_t
            )

            velocity_matrix[i] = np.clip(velocity_matrix[i], -v_max, v_max)

            x_matrix[i] += velocity_matrix[i]*delta_t

            if f(x_matrix[i]) < x_best_values[i]:
                x_best[i] = x_matrix[i].copy()
                x_best_values[i] = f(x_matrix[i])
        x_global_best = x_best[np.argmin(x_best_values)].copy()


    return x_global_best

        
N = 200
n = 2
local_min = []

for i in range(10):
    x_matrix, velocity_matrix = init(N, n, -5, 5)
    best_pos = algo(x_matrix, velocity_matrix, iterations=1000, delta_t=1, vmax=2)
    if not any(np.allclose(best_pos, lm, atol=1e-3) for lm in local_min):
        local_min.append(best_pos)

print(local_min)


p1 = [p[0] for p in local_min]
p2 = [p[1] for p in local_min]
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = np.log(0.01+f([X, Y]))
plt.figure(figsize=(8,6))
contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.scatter(p1, p2, s=100, color="red",zorder=5 )
plt.clabel(contours, inline=True, fontsize=8)
plt.title("Contour Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
