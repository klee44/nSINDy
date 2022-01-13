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
n_forward = 3 # testing
total_steps = 1024 * n_forward # around 30 sec
t = torch.linspace(0, (total_steps)*dt, total_steps+1).to(device)

# system
F = 8
N = 6
    
def lorenz_rhs_torch(t,x):
	f = torch.zeros_like(x)
	for i in range(2,N-1):
		f[:,i] = (x[:,i+1]-x[:,i-2])*x[:,i-1] - x[:,i] + F
	f[:,0]   = (x[:,1] - x[:,N-2]) * x[:,N-1] - x[:,0]   + F
	f[:,1]   = (x[:,2] - x[:,N-1]) * x[:,0]   - x[:,1]   + F
	f[:,N-1] = (x[:,0] - x[:,N-3]) * x[:,N-2] - x[:,N-1] + F
	return f 

# simulation parameters
np.random.seed(2)

# dataset 
n_train = 128
n_val = 128
n_test = 16
n = N 

# simulate training trials 
train_data = np.zeros((n_train, total_steps+1, n))
print('generating training trials ...')
x_init = torch.tensor(np.array([1., 8., 8., 8., 8., 8.])).unsqueeze(0) 

for i in range(n_train):
	x_init_i = x_init + 2*torch.rand(x_init.shape) - 1
	sol = odeint(lorenz_rhs_torch,x_init_i,t,method='dopri5').to(device).squeeze().detach().numpy()
	train_data[i, :, :] = sol
	'''
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(projection='3d')
	ax.plot(train_data[i,:,0],train_data[i,:,1],train_data[i,:,2])
	plt.show()
	'''

# simulate validation trials 
val_data = np.zeros((n_val, total_steps+1, n))
print('generating validation trials ...')
for i in range(n_val):
	x_init_i = x_init + 2*torch.rand(x_init.shape) - 1
	sol = odeint(lorenz_rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	val_data[i, :, :] = sol
    
# simulate test trials
test_data = np.zeros((n_test, total_steps+1, n))
print('generating testing trials ...')
for i in range(n_test):
	x_init_i = x_init + 2*torch.rand(x_init.shape) - 1
	sol = odeint(lorenz_rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	test_data[i, :, :] = sol
	
    
# add noise
train_data += noise*train_data.std(1).mean(0)*np.random.randn(*train_data.shape)
val_data += noise*val_data.std(1).mean(0)*np.random.randn(*val_data.shape)
test_data += noise*test_data.std(1).mean(0)*np.random.randn(*test_data.shape)

np.savez('lorenz96_torch.npz', train_data=train_data,val_data=val_data,test_data=test_data)
