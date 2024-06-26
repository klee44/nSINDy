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
from lib.torchdiffeq import odeint as odeint
#from lib.torchdiffeq import odeint_adjoint as odeint
#import lib.odeint as odeint
import torch.nn.utils.prune as prune
from lib.prune import ThresholdPruning
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

args = parser.parse_args()

torch.set_default_dtype(torch.float64)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

seed = args.r
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

save_path = 'experiments/'
utils.makedirs(save_path)
experimentID = int(SystemRandom().random()*100000)
ckpt_path = os.path.join(save_path, "experiment_" + str(experimentID) + '.ckpt')
fig_save_path = os.path.join(save_path,"experiment_"+str(experimentID))
utils.makedirs(fig_save_path)
print(ckpt_path)

data = np.load("../data/vdp_torch_noise1.npz")
h_ref = 0.01 
Time = 51.20 
N_steps = int(np.floor(Time/h_ref)) + 1
t = np.expand_dims(np.linspace(0,Time,N_steps,endpoint=True,dtype=np.float64),axis=-1)[::1] 
t = torch.tensor(t).squeeze()

'''
train_data = torch.utils.data.DataLoader(torch.tensor(data['train_data']),batch_size=50)
val_data = torch.utils.data.DataLoader(torch.tensor(data['val_data']),batch_size=50)
test_data = torch.utils.data.DataLoader(torch.tensor(data['test_data']),batch_size=50)
'''
train_data = torch.tensor(data['train_data'][:,:,:])
#val_data = torch.tensor(data['val_data'])
#test_data = torch.tensor(data['test_data'])
val_data = torch.utils.data.DataLoader(torch.tensor(data['val_data']),batch_size=50)
test_data = torch.utils.data.DataLoader(torch.tensor(data['test_data']),batch_size=50)
#val_data = torch.utils.data.DataLoader(torch.tensor(data['train_data'][:1,:,:]),batch_size=50)
#test_data = torch.utils.data.DataLoader(torch.tensor(data['train_data'][:1,:,:]),batch_size=50)
odefunc = ODEfuncPoly(2, 3)

parameters_to_prune = ((odefunc.C, "weight"),)

params = odefunc.parameters()
optimizer = optim.Adamax(params, lr=args.lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)

best_loss = 1e30
frame = 0 

for itr in range(args.nepoch):
	print('=={0:d}=='.format(itr))
	for i in range(args.niterbatch):
		optimizer.zero_grad()
		batch_y0, batch_t, batch_y = utils.get_batch(train_data,t,args.lMB,args.nMB)
		pred_y = odeint(odefunc, batch_y0, batch_t, method=args.odeint).to(device).transpose(0,1)
		loss = torch.mean(torch.abs(pred_y - batch_y))
		l1_norm = 1e-4*torch.norm(odefunc.C.weight, p=1)
		loss += l1_norm
		print(itr,i,loss.item(),l1_norm.item())
		loss.backward()
		optimizer.step()
		prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-6)
	scheduler.step()
	
	
	if itr >= 100:
		with torch.no_grad():
			val_loss = 0
			print(odefunc.C.weight)
			
			for d in val_data:
				pred_y = odeint(odefunc, d[:,0,:], t, method=args.odeint).to(device).transpose(0,1)
				val_loss += torch.mean(torch.abs(pred_y - d)).item()
			print('val loss', val_loss)
				
			if best_loss > val_loss:
				print('saving...', val_loss)
				torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)
				best_loss = val_loss 

			plt.figure()
			plt.tight_layout()
			save_file = os.path.join(fig_save_path,"image_{:03d}.png".format(frame))
			fig = plt.figure(figsize=(8,4))
			axes = []
			for i in range(2):
				axes.append(fig.add_subplot(1,2,i+1))
				axes[i].plot(t,d[0,:,i].detach().numpy(),lw=2,color='k')
				axes[i].plot(t,pred_y.detach().numpy()[0,:,i],lw=2,color='c',ls='--')
				plt.savefig(save_file)
			plt.close(fig)
			plt.close('all')
			plt.clf()
			frame += 1


ckpt = torch.load(ckpt_path)
odefunc.load_state_dict(ckpt['state_dict'])

prune.remove(odefunc.C, 'weight')
print(odefunc.C.weight)
torch.save({'state_dict': odefunc.state_dict(),}, ckpt_path)

odefunc.NFE = 0
test_loss = 0
test_sol = np.zeros_like(data['test_data'])
batch_idx = 50
for i, d in enumerate(test_data):
	pred_y = odeint(odefunc, d[:,0,:], t, method=args.odeint).to(device).transpose(0,1)
	test_sol[batch_idx*i:batch_idx*(i+1),:,:] = pred_y.detach().numpy() 
	test_loss += torch.mean(torch.abs(pred_y - d)).item()
print('test loss', test_loss)

fig = plt.figure(figsize=(12,4))
axes = []
for i in range(2):
	axes.append(fig.add_subplot(1,2,i+1))
	axes[i].plot(t,data['test_data'][0,:,i],lw=3,color='k')
	axes[i].plot(t,test_sol[0,:,i],lw=2,color='c',ls='--')

save_file = os.path.join(fig_save_path,"image_best.png")
plt.savefig(save_file)
