import os
import numpy as np
import scipy as sp
from scipy import integrate

import torch
import torch.nn as nn

from torchdiffeq import odeint

torch.set_default_dtype(torch.float64)
np.random.seed(1123)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# adjustable parameters
dt = 0.01       # set to 5e-4 for Lorenz
noise = 0.0      # for study of noisy measurements, we use noise=0.01, 0.02; otherwise we leave it as 0.

n_forwards = np.asarray([7,3,4])

total_steps = 512 * n_forwards
total_step = total_steps.sum()

t = torch.linspace(0, (total_step)*dt, total_step+3).to(device)
t_idx = np.insert(total_steps+1, 0, 0).cumsum()

alphas = torch.tensor(np.random.uniform(.25, 2.5, len(n_forwards))).to(device)
betas = torch.tensor(np.random.uniform(.25, 2.5, len(n_forwards))).to(device)
deltas = torch.tensor(np.random.uniform(.25, 2.5, len(n_forwards))).to(device)
gammas = torch.tensor(np.random.uniform(.25, 2.5, len(n_forwards))).to(device)

print(alphas,betas,deltas,gammas)
# system



# simulation parameters

n = 2

# simulate training trials 
train_data = [] 
x_init = torch.tensor(np.random.uniform(.5, 2.0, n)).to(device).unsqueeze(0)
for i in range(len(n_forwards)):
	def cubic_rhs_torch(t,x):
		return torch.cat( (alphas[i]*x[:,0]-betas[i]*x[:,0]*x[:,1], deltas[i]*x[:,0]*x[:,1]-gammas[i]*x[:,1]), axis=-1)
	sol = odeint(cubic_rhs_torch,x_init,t[t_idx[i]:t_idx[i+1]],method='dopri5').to(device).squeeze().detach().numpy()
	train_data.append(sol)
	print(sol.shape)
	x_init = torch.tensor(sol[-1,:]).to(device).unsqueeze(0)
train_data = np.concatenate(train_data, axis=0)
train_data = np.delete(train_data, total_steps, 0)
print(train_data.shape)
	
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.plot(train_data)
plt.show()
    
# add noise
#train_data += noise*train_data.std(1).mean(0)*np.random.randn(*train_data.shape)

np.savez('lv_torch_{0:d}.npz'.format(len(n_forwards)), train_data=train_data)
