"""
=====================================================================
=====================================================================

Written by: Kyle Daniela Angulo

Program Name: mbe_v2.py

Description: solves the Maxwell-Bloch equations for an electric field with gaussian shape (rabi frequency)

Extra Info: Based on qutip examples and the thesis by Thomas P. Ogden

=====================================================================
=====================================================================
"""

import numpy as np
import matplotlib as mpl 
from matplotlib import pylab as plt
from numpy import linalg
from scipy.special import factorial
import time as time
from scipy import special
from qutip import * 

mpl.rcParams.update({'font.size': 14,'font.family': 'STIXGeneral',
                            'mathtext.fontset': 'cm'})
#constants 
Isat = 3.9e1#Saturation intensity, F=3->F'=4, (isotropic light polarization) in W/m^2
lambda0 = 780e-9#vacuum wavelengthm
c = 3e8#m/s
omega0 = 2*np.pi*c/lambda0
hbar = 1.0545718e-34
gamma = 2*np.pi*6.1e-3 #atomic lifetime in GHz
atomiclifetime = 1/gamma #atomic lifetime in ns
a0 = .52917720859e-10 #Bohr radius in meters
e = 1.602176487e-19 #charge of an electron in Coulombs
matrix_element = 1.956*e*a0 #Effective Dipole Moment F=3->F'=4 (isotropic light polarization) in Coulomb.meters
epsilon0 = 8.854187817e-12 #permittivity of vacuum in farads/meter
g = (2*np.pi/lambda0)*matrix_element**2/(epsilon0*hbar) #light matter coupling constant in Hz and m^2
interactionCrossSection = 2*g/(c*gamma*1e9)

#gaussian envelope
def input_pulse(t,t0,peak,sigma):
	gaussian=peak*np.exp(-(t-t0)**2/2/sigma**2)
	return gaussian

#solver of the Lindblad equation using qutip
#the rabi frequency is time dependent
def solve_Lindblad(t,rabi,rho_0):
	S_12=tensor(basis(2,0)*basis(2,1).dag())
	S_11=tensor(basis(2,0)*basis(2,0).dag())
	S_22=tensor(basis(2,1)*basis(2,1).dag())
	H0=det*S_22
	H1=S_12+S_12.dag()
	H=[H0,[H1,rabi]]
	result = mesolve(H,rho_0,t,np.sqrt(gamma)*S_12,[S_22,S_12.dag()])
	return result

def solve_spatial(rho_z,N_z,rabi_0,dz,g):
	rabi_1=rabi_0+1j*g*(1e3)*dz*rho_z*N_z
	return rabi_1

dt=1.
t_min=-20.
t_max=100.
tlist=np.arange(t_min,t_max,dt)
z_step=200
min_z=0
max_z=400.
dz=1.
zlist=np.arange(min_z,max_z,dz)
# N_z=157.1 #atoms/um
N_z=4.e-2 #atoms/um^3
t0=20
det=0
sigma=10.
peak=gamma/50.
rho_0=basis(2,0)*basis(2,0).dag()
rabi_t=(1 + 1j)*np.zeros([len(tlist),len(zlist)])
rabi_t[:,0]=input_pulse(tlist,t0,peak,sigma)


for i in range(len(zlist)-1):
	time_evolution=solve_Lindblad(tlist,rabi_t[:,i],rho_0)
	s_1=time_evolution.expect[0]
	rho_eg=time_evolution.expect[1]
	rabi_t[:,i+1]=solve_spatial(rho_eg,N_z,rabi_t[:,i],dz,g)

plt.contourf(zlist,tlist,(abs(rabi_t))**2, 50,alpha = 1, cmap = 'viridis')
plt.show()

# data=np.array([tlist,s_1])
# plt.plot(tlist,s_1)
# plt.plot(zlist,spatial_evolution)
# plt.legend(('ground', 'pulse', 'field'),loc='upper right')
# plt.show()
# plt.savefig('mbe.pdf')
# np.savetxt('populations_2level.dat', data.T)