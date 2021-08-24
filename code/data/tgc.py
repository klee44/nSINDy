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
#n_forward = 5*4 # torch_longer
n_forward = 5
total_steps = 1024 * n_forward
t = torch.linspace(0, (total_steps)*dt, total_steps+1).to(device)

# system
# qdot = p/m, 
# pdot = 2/3 ( E_1/q - E_2/(2Lg - q) ) = 2/3 (E_1/q - E_2 (2-q))

# m = 5.0
# Lg = 1

# E1 = exp( 2/3(S1 / Nkb - ln (\hat c q Ac) ) )
# Nkb = 1, Ac = 1
# \hat c = Const

Const = 102.2476703501216
alpha = 1. 
m = 1.0
def rhs_torch(t,x):
	E1 = torch.exp( (2.0/3.0) * ( x[:,2] - torch.log(x[:,0]) - Const ) ) 
	E2 = torch.exp( (2.0/3.0) * ( x[:,3] - torch.log(2.0 - x[:,0]) - Const ) )

	return torch.cat( (x[:,1]/m, (2.0/3.0)*( E1/x[:,0] - E2/(2-x[:,0]) ), 9.0*alpha/(4.0*E1)*(1.0/E1-1.0/E2), - 9.0*alpha/(4.0*E2)*(1.0/E1-1.0/E2)), axis=-1)

S1_init = 1.5*np.log(2.0)+np.log(1)+Const
S2_init = 1.5*np.log(2.0)+np.log(2-1)+Const

# simulation parameters
np.random.seed(2)
n = 4 

# dataset -- GENERIC 
'''
n_train = 320
n_val = 64 
n_test = 64
'''
# dataset -- SINDy 
n_train = 800
n_val = 160
n_test = 160

# simulate training trials 
train_data = np.zeros((n_train, total_steps+1, n))
print('generating training trials ...')
for i in range(n_train):
	x_init = np.random.rand(2)+0.5
	radius = np.random.rand() + .5 
	x_init = np.concatenate((x_init / np.sqrt((x_init**2).sum()) * radius, [S1_init, S2_init]))
	#x_init = [1,2,S1_init,S2_init]
	x_init = torch.tensor(x_init).to(device).unsqueeze(0)
	sol = odeint(rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	train_data[i, :, :] = sol
	'''
	import matplotlib.pyplot as plt
	plt.plot(sol[:,0],'r')
	plt.plot(sol[:,1],'b')
	#plt.plot(sol[:,2],'g')
	plt.show()
	'''
# simulate validation trials 
val_data = np.zeros((n_val, total_steps+1, n))
print('generating validation trials ...')
for i in range(n_val):
	x_init = np.random.rand(2)+0.5	
	radius = np.random.rand() + .5
	x_init = np.concatenate((x_init / np.sqrt((x_init**2).sum()) * radius, [S1_init, S2_init]))
	x_init = torch.tensor(x_init).to(device).unsqueeze(0)
	sol = odeint(rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	val_data[i, :, :] = sol
    
# simulate test trials
test_data = np.zeros((n_test, total_steps+1, n))
print('generating testing trials ...')
for i in range(n_test):
	x_init = np.random.rand(2)+0.5
	radius = np.random.rand() + .5
	x_init = np.concatenate((x_init / np.sqrt((x_init**2).sum()) * radius, [S1_init, S2_init]))
	x_init = torch.tensor(x_init).to(device).unsqueeze(0)
	sol = odeint(rhs_torch,x_init,t,method='dopri5').to(device).squeeze().detach().numpy()
	test_data[i, :, :] = sol
    
# add noise
train_data += noise*train_data.std(1).mean(0)*np.random.randn(*train_data.shape)
val_data += noise*val_data.std(1).mean(0)*np.random.randn(*val_data.shape)
test_data += noise*test_data.std(1).mean(0)*np.random.randn(*test_data.shape)

np.savez('tgc_torch_sindy.npz', train_data=train_data,val_data=val_data,test_data=test_data)
