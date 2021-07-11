import numpy as np
import torch
import torch.nn as nn

import lib.utils as utils
from lib.utils import TensorProduct, Taylor

#####################################################################################################

class ODEFunc_(nn.Module):
	def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc, self).__init__()

		self.input_dim = input_dim
		self.device = device

		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)

class ODEfunc(nn.Module):
	def __init__(self, dim, nlayer, nunit, device = torch.device("cpu")):
		super(ODEfunc, self).__init__()
		self.gradient_net = utils.create_net(dim, dim, n_layers=nlayer, n_units=nunit, nonlinear = nn.Tanh).to(device)
		self.NFE = 0

	def forward(self, t, y):
		output = self.gradient_net(y)
		return output 

class ODEfuncPoly(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncPoly, self).__init__()
		self.NFE = 0
		#self.TP = TensorProduct(dim,order)
		self.TP = Taylor(dim,order)
		#self.C = nn.Parameter(torch.randn((self.TP.nterms, dim), requires_grad=True))
		self.C = nn.Linear(self.TP.nterms,dim,bias=False)

	def forward(self, t, y):
		P = self.TP(y)
		#output = torch.einsum('ab,za->zb',self.C,P)
		output = self.C(P)
		return output 
