import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os, sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import random
from random import SystemRandom

import matplotlib.pyplot as plt

import lib.utils as utils
from lib.odefunc import ODEfunc, ODEfuncGNN, ODEfuncPolyTrig
from lib.torchdiffeq import odeint as odeint
#from lib.torchdiffeq import odeint_adjoint as odeint
#import lib.odeint as odeint

import argparse
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--r', type=int, default=0, help='random_seed')

parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--nepoch', type=int, default=100, help='max epochs')
parser.add_argument('--niterbatch', type=int, default=100, help='max epochs')

parser.add_argument('--nlayer', type=int, default=4, help='max epochs')
parser.add_argument('--nunit', type=int, default=25, help='max epochs')

parser.add_argument('--lMB', type=int, default=100, help='length of seq in each MB')
parser.add_argument('--nMB', type=int, default=40, help='length of seq in each MB')

parser.add_argument('--odeint', type=str, default='rk4', help='integrator')
parser.add_argument('--id', type=int, default=0, help='exp id')
parser.add_argument('--id2', type=int, default=0, help='exp id')

args = parser.parse_args()

torch.set_default_dtype(torch.float64)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

seed = args.r
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

save_path = 'experiments/'
utils.makedirs(save_path)
experimentID = args.id 
ckpt_path = os.path.join(save_path, "experiment_" + str(experimentID) + '.ckpt')
fig_save_path = os.path.join(save_path,"experiment_"+str(experimentID))
utils.makedirs(fig_save_path)

experimentID2 = args.id2 
ckpt_path2 = os.path.join(save_path, "experiment_" + str(experimentID2) + '.ckpt')
print(ckpt_path, ckpt_path2)

data = np.load("../data/dno_torch.npz")
#data = np.load("../data/dno_torch_timeunit100.npz")
h_ref = 0.001 
Time = 5.120 
#Time = 102.4
N_steps = int(np.floor(Time/h_ref)) + 1
t = np.expand_dims(np.linspace(0,Time,N_steps,endpoint=True,dtype=np.float64),axis=-1)[::1] 
t = torch.tensor(t).squeeze()

odefunc = ODEfuncGNN(3, 3, 1, 1)

ckpt = torch.load(ckpt_path)
odefunc.load_state_dict(ckpt['state_dict'])

test_sol = np.zeros((10,len(t),3))
d = torch.tensor(data['test_data'][:10],requires_grad=True)
pred_y = odeint(odefunc, d[:,0,:], t, method=args.odeint).to(device).transpose(0,1)
test_sol = pred_y.detach().numpy() 

odefunc2 = ODEfuncPolyTrig(3, 3)

ckpt2 = torch.load(ckpt_path2)
odefunc2.load_state_dict(ckpt2['state_dict'])

test_sol2 = np.zeros((10,len(t),3))
d = torch.tensor(data['test_data'][:10],requires_grad=True)
pred_y = odeint(odefunc2, d[:,0,:], t, method=args.odeint).to(device).transpose(0,1)
test_sol2 = pred_y.detach().numpy() 

target_id = 7 


dxdt = odefunc(t,torch.tensor(test_sol[target_id,:,:],requires_grad=True))
dxdt2 = odefunc2(t,torch.tensor(test_sol2[target_id,:,:],requires_grad=True))
print((dxdt.detach().numpy()[:,2]<0).any(),(dxdt2.detach().numpy()[:,2]<0).any())

fig = plt.figure(figsize=(5.5,2.))

plt.axhline(y=0.0, color='k', linestyle='-', lw=2)
plt.plot(t,dxdt.detach().numpy()[:,2],lw=2,color='b',ls='-')
plt.plot(t,dxdt2.detach().numpy()[:,2],lw=2,color='r',ls='--')

plt.margins(0,0.04)
plt.title('Damped nonlinear oscillator - dSdt')
#plt.tight_layout()

save_file = os.path.join(fig_save_path,"dno_dSdt_all_example_tu100.png")
#save_file = os.path.join(fig_save_path,"dno_dSdt_all_example.png")
plt.savefig(save_file)
plt.close(fig)
plt.close('all')
plt.clf()

fig = plt.figure(figsize=(5.5,2.))

x_ref = data['test_data'][target_id,:,:]
x_aprx= test_sol[target_id,:,:]
x_aprx2= test_sol2[target_id,:,:]

E_ref = x_ref[:,1]**2/2.0 - 6.0*np.cos(x_ref[:,0]) + x_ref[:,2]
E_aprx= x_aprx[:,1]**2/2.0 - 6.0*np.cos(x_aprx[:,0]) + x_aprx[:,2]
E_aprx2= x_aprx2[:,1]**2/2.0 - 6.0*np.cos(x_aprx2[:,0]) + x_aprx2[:,2]

plt.plot(t,E_ref,lw=2,color='k')
plt.plot(t,E_aprx,lw=2,color='b',ls='-')
plt.plot(t,E_aprx2,lw=2,color='r',ls='--')

plt.margins(0,0.04)
plt.title('Damped nonlinear oscillator - E')
#plt.tight_layout()

save_file = os.path.join(fig_save_path,"dno_E_all_example_tu100.png")
#save_file = os.path.join(fig_save_path,"dno_E_all_example.png")
plt.savefig(save_file)
plt.close(fig)
plt.close('all')
plt.clf()
