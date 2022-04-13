import os
import numpy as np
import scipy as sp
from scipy import integrate

import torch
import torch.nn as nn

from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# adjustable parameters
dt = 5e-4       # set to 5e-4 for Lorenz
noise = 0#10      # for study of noisy measurements, we use noise=0.01, 0.02; otherwise we leave it as 0.
n_forward = 5
#n_forward = 30 # testing
total_steps = 1024 * n_forward
t = torch.linspace(0, (total_steps)*dt, total_steps+1).to(device)

# system
sigma = 10
rho = 28
beta = 8/3
    
def lorenz_rhs(x):
	return np.array([sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]])

def lorenz_rhs_torch(t,x):
	return torch.cat((sigma*(x[:,1]-x[:,0]),x[:,0]*(rho-x[:,2])-x[:,1],x[:,0]*x[:,1]-beta*x[:,2]),-1)

# simulation parameters
np.random.seed(2)
warmup = 10000
n = 3

# dataset 
n_train = 10#80#1600
n_val = 10#16#320
n_test = 10#16#320

# simulate training trials 
pre_t = np.linspace(0, warmup*dt, warmup+1)

train_data = np.zeros((n_train, total_steps+1, n))
print('generating training trials ...')
x_init = np.random.uniform(-0.1, 0.1, n)
sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, warmup*dt], x_init, t_eval=pre_t)
sol = sol.y.T

for i in range(n_train):
	x_init = torch.tensor(sol[-2:-1, :]).to(device)
	sol = odeint(lorenz_rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	#train_data[i, :, :] = sol
	NoiseMag=[np.std(sol[:,k])*noise*0.01 for k in range(n)]
	train_data[i, :, :] = sol + np.hstack([NoiseMag[k]*np.random.randn(total_steps+1,1) for k in range(n)])

# simulate validation trials 
val_data = np.zeros((n_val, total_steps+1, n))
print('generating validation trials ...')
x_init = np.random.uniform(-0.1, 0.1, n)
sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, warmup*dt], x_init, t_eval=pre_t)
sol = sol.y.T

for i in range(n_val):
	x_init = torch.tensor(sol[-2:-1, :]).to(device)
	sol = odeint(lorenz_rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	#val_data[i, :, :] = sol
	NoiseMag=[np.std(sol[:,k])*noise*0.01 for k in range(n)]
	val_data[i, :, :] = sol + np.hstack([NoiseMag[k]*np.random.randn(total_steps+1,1) for k in range(n)])

    
# simulate test trials
test_data = np.zeros((n_test, total_steps+1, n))
print('generating testing trials ...')
x_init = np.random.uniform(-0.1, 0.1, n)
sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(x), [0, warmup*dt], x_init, t_eval=pre_t)
sol = sol.y.T

for i in range(n_test):
	x_init = torch.tensor(sol[-2:-1, :]).to(device)
	sol = odeint(lorenz_rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	test_data[i, :, :] = sol
	
    
# add noise
#train_data += noise*train_data.std(1).mean(0)*np.random.randn(*train_data.shape)
#val_data += noise*val_data.std(1).mean(0)*np.random.randn(*val_data.shape)
#test_data += noise*test_data.std(1).mean(0)*np.random.randn(*test_data.shape)

np.savez('lorenz_torch_si.npz', train_data=train_data,val_data=val_data,test_data=test_data)
