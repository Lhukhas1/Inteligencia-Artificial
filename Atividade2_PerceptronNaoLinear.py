#Atividade 2 - Otimização contínua

#Aluno: Lucas Severino da Silva   

# -*- coding: utf-8 -*-
"""
Exemplo de perceptron não-linear para funções lógicas

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import basinhopping, differential_evolution



# Heaviside step function
@np.vectorize
def heaviside(x):
    return 1.0 if x > 0.0 else 0.0

# Relu step (activation) function
@np.vectorize
def relu(x):
    return x * (x > 0)

# helper function for decision boundary plotting
def perceptron(x, w):
    #return heaviside(np.sum(np.concatenate((x,[1])) * w))
    return relu(np.sum(np.concatenate((x,[1])) * w)) # TODO: delete line above (27) and uncomment this one, then change train() function

def xorperceptron3d(x, y, pand, pnand, por):
    y1 = perceptron([x,y], por)
    y2 = perceptron([x,y], pnand)
    return perceptron([y1, y2], pand)

# Calculate MSE (mean squared error) for a training data set
def loss(x, y, w):
    mse = 0
    n = len(y)
    
    for input_val, output in zip(x, y):
        error = xorperceptron3d(input_val[0], input_val[1], w[0:3], w[3:6], w[6:]) - output
        mse += math.pow(error, 2)/n
        
    return mse

# Training data for XOR 
x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([0, 1, 1, 0])






def objective_function(w):
    return loss(x_train, y_train, w)

# Quase sempre as fronteria de decisao estavam erradas entao decidi tentar
# com mais de um local aleatorio e pegar o melhor pra plotar
def train_basin_hopping():
    print("\nBasin Hopping")
    results = []   
    for i in range(5):  
        x0 = np.random.uniform(-1, 2, 9)
        
        minimizer_kwargs = {"method": "BFGS"}
        
        result = basinhopping(objective_function, x0, minimizer_kwargs=minimizer_kwargs, niter=1000)
        
        result_i = {
            'weights': result.x,
            'mse': result.fun,
        }
        results.append(result_i)
        
        if result.fun < 0.00001 :
            break
        
    results.sort(key=lambda x: x['mse'])
    
    best = results[0]
    print(f"MSE final: {best['mse']:.6f}")
    
    return best['weights']


def train_differential_evolution():
    print("\nDifferential Evolution")
    
    bounds = [(-10, 10)] * 9 
    
    result = differential_evolution(objective_function, bounds, maxiter = 1000, popsize = 15,
                                    tol = 1e-6, atol=1e-8)
    
    print(f"MSE final: {result.fun:.6f}")
    
    return result.x

def plot_results(w_list, method_names):
    for idx, (w, method_name) in enumerate(zip(w_list, method_names)):
        if w is None:
            continue
            
        pand = w[0:3]
        pnand = w[3:6]
        por = w[6:]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_train[:,0], x_train[:,1], y_train, c='b', marker='o', s=100)
        
        xx1 = np.linspace(0, 1, 100)
        xx2 = np.linspace(0, 1, 100)
        x_mesh, y_mesh = np.meshgrid(xx1, xx2)
        z = np.vectorize(xorperceptron3d, excluded=['pand', 'pnand', 'por'])(x=x_mesh, y=y_mesh,
                                                                             pand=pand, pnand=pnand, por=por)
        
        zlow = np.copy(z)
        zhigh = np.copy(z)
        zlow[z<=0] = np.nan
        zhigh[z>0] = np.nan
        ax.plot_surface(x_mesh, y_mesh, zlow, color='r', alpha=0.2)
        ax.plot_surface(x_mesh, y_mesh, zhigh, color='b', alpha=0.2)
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title(f'{method_name}')
        ax.view_init(90, 0)  
        plt.show()


if __name__ == "__main__":
    weights_list = []
    method_names = []
    
    try:
        w_bh = train_basin_hopping()
        weights_list.append(w_bh)
        method_names.append("Basin-Hopping")
    except Exception as e:
        print(f"Erro no Basin-Hopping: {e}")
    
    try:
        w_de = train_differential_evolution()
        weights_list.append(w_de)
        method_names.append("Differential Evolution")
    except Exception as e:
        print(f"Erro no Differential Evolution: {e}")

    plot_results(weights_list, method_names)