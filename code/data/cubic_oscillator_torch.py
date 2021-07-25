import os
import numpy as np
import scipy as sp
from scipy import integrate

import torch
import torch.nn as nn

from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# adjustable parameters
dt = 0.01       # set to 5e-4 for Lorenz
noise = 0.      # for study of noisy measurements, we use noise=0.01, 0.02; otherwise we leave it as 0.
n_forward = 5
total_steps = 1024 * n_forward
t = torch.linspace(0, (total_steps)*dt, total_steps+1).to(device)

# system
def cubic_rhs(x):
	return np.array([-0.1*x[0]**3+2*x[1]**3, -2*x[0]**3-0.1*x[1]**3])

def cubic_rhs_torch(t,x):
	return torch.cat( (-0.1*x[:,0]**3+2.0*x[:,1]**3, -2.0*x[:,0]**3-0.1*x[:,1]**3), axis=-1)


# simulation parameters
np.random.seed(2)
n = 2

# dataset 
n_train = 1600
n_val = 320
n_test = 320

# simulate training trials 
train_data = np.zeros((n_train, total_steps+1, n))
print('generating training trials ...')
for i in range(n_train):
	x_init = torch.tensor(np.random.uniform(-1.0, 1.0, n)).to(device).unsqueeze(0)
	sol = odeint(cubic_rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	train_data[i, :, :] = sol
	

# simulate validation trials 
val_data = np.zeros((n_val, total_steps+1, n))
print('generating validation trials ...')
for i in range(n_val):
	x_init = torch.tensor(np.random.uniform(-1.0, 1.0, n)).to(device).unsqueeze(0)
	sol = odeint(cubic_rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	val_data[i, :, :] = sol
    
# simulate test trials
test_data = np.zeros((n_test, total_steps+1, n))
print('generating testing trials ...')
for i in range(n_test):
	x_init = torch.tensor(np.random.uniform(-1.0, 1.0, n)).to(device).unsqueeze(0)
	sol = odeint(cubic_rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	test_data[i, :, :] = sol
    
# add noise
train_data += noise*train_data.std(1).mean(0)*np.random.randn(*train_data.shape)
val_data += noise*val_data.std(1).mean(0)*np.random.randn(*val_data.shape)
test_data += noise*test_data.std(1).mean(0)*np.random.randn(*test_data.shape)

np.savez('cubic_oscillator_torch.npz', train_data=train_data,val_data=val_data,test_data=test_data)
