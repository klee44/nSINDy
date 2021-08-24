import os
import numpy as np
import scipy as sp
from scipy import integrate

import torch
import torch.nn as nn

from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# adjustable parameters
dt = 0.001       # set to 5e-4 for Lorenz
noise = 0.      # for study of noisy measurements, we use noise=0.01, 0.02; otherwise we leave it as 0.
n_forward = 5  # training
n_forward = 100 # testing
total_steps = 1024 * n_forward
t = torch.linspace(0, (total_steps)*dt, total_steps+1).to(device)

# system
# m = T = 1
# gamma = .25
# k = 6.
def rhs_torch(t,x):
	return torch.cat( (x[:,1], -6.*torch.sin(x[:,0])-0.25*x[:,1], 0.25*x[:,1]**2), axis=-1)

# simulation parameters
np.random.seed(2)
n = 3 

# dataset 
n_train = 800
n_val = 160 
n_test = 160

odeint_method = 'dopri5'

# simulate training trials 
train_data = np.zeros((n_train, total_steps+1, n))
print('generating training trials ...')
for i in range(n_train):
	x_init = np.random.rand(2)*2.-1	
	radius = np.random.rand() + 1.3
	x_init = np.concatenate((x_init / np.sqrt((x_init**2).sum()) * radius, [0]))
	x_init = torch.tensor(x_init).to(device).unsqueeze(0)
	sol = odeint(rhs_torch,x_init,t,method=odeint_method).to(device).squeeze().detach().numpy()
	train_data[i, :, :] = sol
	'''
	import matplotlib.pyplot as plt
	plt.plot(sol[:,0],'r')
	plt.plot(sol[:,1],'b')
	plt.plot(sol[:,2],'g')
	plt.show()
	'''

# simulate validation trials 
val_data = np.zeros((n_val, total_steps+1, n))
print('generating validation trials ...')
for i in range(n_val):
	x_init = np.random.rand(2)*2.-1	
	radius = np.random.rand() + 1.3
	x_init = np.concatenate((x_init / np.sqrt((x_init**2).sum()) * radius, [0]))
	x_init = torch.tensor(x_init).to(device).unsqueeze(0)
	sol = odeint(rhs_torch,x_init,t,method=odeint_method).to(device).squeeze().detach().numpy()
	val_data[i, :, :] = sol
    
# simulate test trials
test_data = np.zeros((n_test, total_steps+1, n))
print('generating testing trials ...')
for i in range(n_test):
	x_init = np.random.rand(2)*2.-1	
	radius = np.random.rand() + 1.3
	x_init = np.concatenate((x_init / np.sqrt((x_init**2).sum()) * radius, [0]))
	x_init = torch.tensor(x_init).to(device).unsqueeze(0)
	sol = odeint(rhs_torch,x_init,t,method=odeint_method).to(device).squeeze().detach().numpy()
	test_data[i, :, :] = sol
    
# add noise
train_data += noise*train_data.std(1).mean(0)*np.random.randn(*train_data.shape)
val_data += noise*val_data.std(1).mean(0)*np.random.randn(*val_data.shape)
test_data += noise*test_data.std(1).mean(0)*np.random.randn(*test_data.shape)

np.savez('dno_torch_timeunit100.npz', train_data=train_data,val_data=val_data,test_data=test_data)
