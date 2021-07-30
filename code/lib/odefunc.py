import numpy as np
import torch
import torch.nn as nn

import lib.utils as utils
from lib.utils import TensorProduct, Taylor, TotalDegree, TotalDegreeTrig

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
		self.TP = TotalDegree(dim,order)
		#self.TP = Taylor(dim,order)
		#self.C = nn.Parameter(torch.randn((self.TP.nterms, dim), requires_grad=True))
		self.C = nn.Linear(self.TP.nterms,dim,bias=False)

	def forward(self, t, y):
		P = self.TP(y)
		#output = torch.einsum('ab,za->zb',self.C,P)
		output = self.C(P)
		return output 

class ODEfuncPolyTrig(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncPolyTrig, self).__init__()
		self.NFE = 0
		self.TP = TotalDegreeTrig(dim,order)
		self.C = nn.Linear(self.TP.nterms,dim,bias=False)

	def forward(self, t, y):
		P = self.TP(y)
		output = self.C(P)
		return output

class ODEfuncHNN(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncHNN, self).__init__()
		self.NFE = 0
		self.TP = TotalDegree(dim,order)
		self.C = nn.Linear(self.TP.nterms,1,bias=False)
		self.L = np.zeros((2,2))
		self.L[0,1], self.L[1,0] = 1, -1
		self.L = torch.tensor(self.L).to(device)

	def forward(self, t, y):
		P = self.TP(y)
		H = self.C(P) 

		dH = torch.autograd.grad(H.sum(), y, create_graph=True)[0]
		output = dH @ self.L.t()
		return output 

class ODEfuncHNNTrig(nn.Module):
	def __init__(self, dim, order, device = torch.device("cpu")):
		super(ODEfuncHNNTrig, self).__init__()
		self.NFE = 0
		self.TP = TotalDegreeTrig(dim,order)
		self.C = nn.Linear(self.TP.nterms,1,bias=False)
		self.L = np.zeros((2,2))
		self.L[0,1], self.L[1,0] = 1, -1
		self.L = torch.tensor(self.L).to(device)

	def forward(self, t, y):
		P = self.TP(y)
		H = self.C(P) 

		dH = torch.autograd.grad(H.sum(), y, create_graph=True)[0]
		output = dH @ self.L.t()
		return output 

class ODEfuncGNN(nn.Module):
	def __init__(self, dim, order, D1, D2, device = torch.device("cpu")):
		super(ODEfuncGNN, self).__init__()
		self.NFE = 0

		self.P_E = TotalDegreeTrig(dim,order)
		self.C = nn.Linear(self.P_E.nterms,1,bias=False)

		self.L = np.zeros((3,3))
		self.L[0,1], self.L[1,0] = 1, -1
		self.L = torch.tensor(self.L).to(device)

		self.D_M = nn.Parameter(torch.randn((D1, D2), requires_grad=True))
		self.L_M = nn.Parameter(torch.randn((dim, dim, D1), requires_grad=True))

	def friction_matrix(self,dE):
		D = self.D_M @ torch.transpose(self.D_M, 0, 1)
		L = (self.L_M - torch.transpose(self.L_M, 0, 1))/2.0
		zeta = torch.einsum('abm,mn,cdn->abcd',L,D,L) # zeta [alpha, beta, mu, nu] 
		self.M = torch.einsum('abmn,zb,zn->zam',zeta,dE,dE)

	def friction_matvec(self,dE,dS): 	
		D = self.D_M @ torch.transpose(self.D_M, 0, 1)
		L = (self.L_M - torch.transpose(self.L_M, 0, 1))/2.0
		zeta = torch.einsum('abm,mn,cdn->abcd',L,D,L) # zeta [alpha, beta, mu, nu] 
		MdS = torch.einsum('abmn,zb,zm,zn->za',zeta,dE,dS,dE)
		return MdS 

	def forward(self, t, y):
		P_E = self.P_E(y)
		E = self.C(P_E) 

		dE = torch.autograd.grad(E.sum(), y, create_graph=True)[0]
		LdE = dE @ self.L.t()

		S = y[:,-1]
		dS = torch.autograd.grad(S.sum(), y, create_graph=True)[0]

		MdS = self.friction_matvec(dE,dS)
		output = LdE + MdS

		self.friction_matrix(dE) 
		return output 
