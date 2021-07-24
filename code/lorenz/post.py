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
from lib.odefunc import ODEfunc, ODEfuncPoly
#from lib.torchdiffeq import odeint as odeint
from lib.torchdiffeq import odeint_adjoint as odeint
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
print(ckpt_path)

data = np.load("../data/lorenz_torch.npz")
h_ref = 5e-4 
Time = 2.56 
N_steps = int(np.floor(Time/h_ref)) + 1
t = np.expand_dims(np.linspace(0,Time,N_steps,endpoint=True,dtype=np.float64),axis=-1)[::1] 
t = torch.tensor(t).squeeze()



test_data = torch.utils.data.DataLoader(torch.tensor(data['test_data']),batch_size=50)
odefunc = ODEfunc(3, args.nlayer, args.nunit)
#odefunc = ODEfuncPoly(3, 2)

ckpt = torch.load(ckpt_path)
odefunc.load_state_dict(ckpt['state_dict'])
#print(odefunc.C.weight.detach().numpy())

odefunc.NFE = 0
test_loss = 0
test_sol = np.zeros_like(data['test_data'])
batch_idx = 50
for i, d in enumerate(test_data):
	pred_y = odeint(odefunc, d[:,0,:], t, method=args.odeint).to(device).transpose(0,1)
	test_sol[batch_idx*i:batch_idx*(i+1),:,:] = pred_y.detach().numpy() 
	test_loss += torch.mean(torch.abs(pred_y - d)).item()
print('test loss', test_loss)

for i in range(10):
	fig = plt.figure(figsize=(8,4))
	axes = []

	axes.append(fig.add_subplot(1,2,1))
	for k in range(3):
		axes[0].plot(t,data['test_data'][i,:,k],lw=3,color='k')
		axes[0].plot(t,test_sol[i,:,k],lw=2,color='c',ls='--')

	axes.append(fig.add_subplot(1,2,2))
	err = np.sum(np.power(data['test_data'][i,:,:] - test_sol[i,:,:],2),axis=1) 
	axes[1].plot(t[1:],np.log10(err[1:]))

	save_file = os.path.join(fig_save_path,"image_best_{}.png".format(i))
	plt.savefig(save_file)
	plt.close(fig)
	plt.close('all')
	plt.clf()
