import math
import numpy as np # optional - uncomment and use if you wish
import matplotlib.pyplot as plt

# ==============================
# run_gradient_descent function:
# ==============================

def run_gradient_descent(x_start, mu, eta, gradient_tolerance):

  x_list = [x_start]
  tolerance = math.inf 

  while tolerance > gradient_tolerance:
    gradient = compute_gradient(x_list[-1],mu)
    x_step = [x_list[-1][0] - eta*gradient[0], x_list[-1][1] - eta*gradient[1]]
    x_list.append(x_step)

    # L2 norm
    tolerance = ((x_list[-1][0] - x_list[-2][0])**2 + (x_list[-1][1]-x_list[-2][1])**2)**0.5

  return(x_list[-1])

# ==============================
# compute_gradient function:
# ==============================

def compute_gradient(x, mu):

  gradient = [0,0]  
  gradient[0] = (2*(x[0]-1) + mu*4*x[0]*(x[0]**2+x[1]**2-1)) 
  gradient[1]= (4*(x[1]-2) + mu*4*x[1]*(x[0]**2+x[1]**2-1))

  return(gradient)

# ==============================
# Main program:
# ==============================

mu_values = [1, 10, 100, 1000]
eta = 0.0001
x_start = [1,2]
gradient_tolerance = 0.0000001

mu_list = []
x0_list = []
x1_list = []

for mu in mu_values:
  x = run_gradient_descent(x_start, mu, eta, gradient_tolerance)
  output = f"x = ({x[0]:.4f}, {x[1]:.4f}), mu = {mu:.1f}"
  mu_list.append(mu)
  x0_list.append(x[0])
  x1_list.append(x[1])
  print(output)


plt.figure(figsize=(10, 6))
plt.plot(mu_list, x0_list, label='x[0]', marker='o')
plt.plot(mu_list, x1_list, label='x[1]', marker='s')
plt.xlabel('mu')
plt.ylabel('x values')
plt.title('Gradient Descent Result vs mu')
plt.legend()
plt.grid(True)
plt.show()