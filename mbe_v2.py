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
Isat = 3.1e1#Saturation intensity, F=3->F'=4, (isotropic light polarization) in W/m^2
lambda0 = 780e-9#vacuum wavelengthm
c = 3e8#m/s
omega0 = 2*np.pi*c/lambda0
hbar = 1.0545718e-34
atomiclifetime = 26.5 #atomic lifetime in ns
gamma = 1/(atomiclifetime) #atomic lifetime in GHz
a0 = .52917720859e-10 #Bohr radius in meters
e = 1.602176487e-19 #charge of an electron in Coulombs
matrix_element = 1.956*e*a0/np.sqrt(.8068842295321867) #Effective Dipole Moment F=3->F'=4 (isotropic light polarization) in Coulomb.meters
epsilon0 = 8.854187817e-12 #permittivity of vacuum in farads/meter
g =  (2*np.pi/lambda0)*matrix_element**2/(epsilon0*hbar) #light matter coupling constant
interactionCrossSection = 2*g/(c*gamma*1e9)

#gaussian envelope
def input_pulse(t,t0,sigma,det):
	gaussian=3.*np.exp(-(t-t0)**2/2/sigma**2)
	return gaussian

#solver of the Lindblad equation using qutip
#the rabi frequency is time dependent
def solve_Lindblad(t,rabi,rho_0):
	gamma=6.
	S_12=tensor(basis(2,0)*basis(2,1).dag())
	S_11=tensor(basis(2,0)*basis(2,0).dag())
	S_22=tensor(basis(2,1)*basis(2,1).dag())
	H0=det*S_22
	H1=S_12+S_12.dag()
	H=[H0,[H1,rabi]]
	result = mesolve(H,rho_0,t,np.sqrt(gamma)*S_12,[S_22,S_12.dag()])
	return result

def solve_spatial(rho_z,N_z,rabi_0,z_steps,dz):
	rabi=np.zeros(z_steps)
	rabi[0]=rabi_0
	for i in range(z_steps-1):
		rabi[i+1]=rabi[i]+1j*dz*rho_z
	return rabi

t_steps=200
tlist=np.linspace(-20.,20.0, t_steps)
z_steps=200
min_z=0
max_z=20.
dz=(max_z-min_z)/z_steps
zlist=np.linspace(min_z,max_z, z_steps)
t0=0.
sigma=3.
det=0
rho_0=basis(2,0)*basis(2,0).dag()
rabi_0=input_pulse(tlist,t0,sigma,det)
time_evolution=solve_Lindblad(tlist,rabi_0,rho_0)
s_1=time_evolution.expect[0]
rho_eg=time_evolution.expect[1]
N_z=1.
spatial_evolution=solve_spatial(rho_eg[100],N_z,rabi_0[100],z_steps,dz)

data=np.array([tlist,s_1])
# plt.plot(tlist,s_1)
plt.plot(zlist,spatial_evolution)
plt.legend(('ground', 'pulse', 'field'),loc='upper right')
plt.show()
# plt.savefig('mbe.pdf')
# np.savetxt('populations_2level.dat', data.T)