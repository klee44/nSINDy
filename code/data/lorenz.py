import os
import numpy as np
import scipy as sp
from scipy import integrate

# adjustable parameters
dt = 5e-4       # set to 5e-4 for Lorenz
noise = 0.      # for study of noisy measurements, we use noise=0.01, 0.02; otherwise we leave it as 0.
n_forward = 5
total_steps = 10240 * n_forward
t = np.linspace(0, (total_steps)*dt, total_steps+1)

# system
sigma = 10
rho = 28
beta = 8/3
    
def lorenz_rhs(x):
	return np.array([sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]])

# simulation parameters
np.random.seed(2)
warmup = 10000
n = 3

# dataset 
n_train = 1600
n_val = 320
n_test = 320

# simulate training trials 
pre_t = np.linspace(0, warmup*dt, warmup+1)

train_data = np.zeros((n_train, total_steps+1, n))
print('generating training trials ...')
x_init = np.random.uniform(-0.1, 0.1, n)
sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, warmup*dt], x_init, t_eval=pre_t)
for i in range(n_train):
	x_init = sol.y[:, -1].T
	sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, total_steps*dt], x_init, t_eval=t)
	train_data[i, :, :] = sol.y.T

# simulate validation trials 
val_data = np.zeros((n_val, total_steps+1, n))
print('generating validation trials ...')
x_init = np.random.uniform(-0.1, 0.1, n)
sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, warmup*dt], x_init, t_eval=pre_t)
for i in range(n_val):
	x_init = sol.y[:, -1].T    
	sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, total_steps*dt], x_init, t_eval=t)
	val_data[i, :, :] = sol.y.T
    
# simulate test trials
test_data = np.zeros((n_test, total_steps+1, n))
print('generating testing trials ...')
x_init = np.random.uniform(-0.1, 0.1, n)
sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, warmup*dt], x_init, t_eval=pre_t)
for i in range(n_test):
	x_init = sol.y[:, -1].T
	sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, total_steps*dt], x_init, t_eval=t)
	test_data[i, :, :] = sol.y.T
    
# add noise
train_data += noise*train_data.std(1).mean(0)*np.random.randn(*train_data.shape)
val_data += noise*val_data.std(1).mean(0)*np.random.randn(*val_data.shape)
test_data += noise*test_data.std(1).mean(0)*np.random.randn(*test_data.shape)

np.savez('lorenz_extra.npz', train_data=train_data,val_data=val_data,test_data=test_data)
