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


mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 14})

#constants 
Isat = 3.9e1#Saturation intensity, F=3->F'=4, (isotropic light polarization) in W/m^2
lambda0 = 780e-9#vacuum wavelengthm
c = 3e8 #m/s
omega0 = 2*np.pi*c/lambda0
hbar = 1.0545718e-34
gamma = 2*np.pi*6.1e-3 #atomic lifetime in GHz
atomiclifetime = 1/gamma #atomic lifetime in ns
a0 = .52917720859e-10 #Bohr radius in meters
e = 1.602176487e-19 #charge of an electron in Coulombs
matrix_element = 1.956*e*a0 #Effective Dipole Moment F=3->F'=4 (isotropic light polarization) in Coulomb meters
epsilon0 = 8.854187817e-12 #permittivity of vacuum in farads/meter
g = (2*np.pi/lambda0)*matrix_element**2/(epsilon0*hbar) #light matter coupling constant in Hz and m^2
sigma_0=1.246e-9 #resonant cross section cm^2

#gaussian envelope
def input_pulse(t,t0,peak,sigma):
	gaussian=peak*np.exp(-(t-t0)**2/2/sigma**2)
	return gaussian

#solver of the Lindblad equation using qutip, the rabi frequency is time dependent
def solve_Lindblad(t,rabi,rho_0):
	S_12=tensor(basis(2,0)*basis(2,1).dag())
	S_11=tensor(basis(2,0)*basis(2,0).dag())
	S_22=tensor(basis(2,1)*basis(2,1).dag())
	H0=det*S_22
	H1=S_12+S_12.dag()
	H=[H0,[H1,rabi]]
	result = mesolve(H,rho_0,t,np.sqrt(gamma)*S_12,[S_22,S_12.dag()])
	return result

#one step of spatial evolution or paraxial equation
def solve_spatial(rho_z,N_z,rabi_0,dz,g):
	rabi_1=rabi_0+1j*g*(1e3)*dz*rho_z*N_z
	return rabi_1

dt=0.1
t_min=-20.
t_max=100.
tlist=np.arange(t_min,t_max,dt)
dz=5.
min_z=0
max_z=400.
zlist=np.arange(min_z,max_z,dz)
N_z=8.e-2 #atoms/um^3
t0=20
det=0
width=0.5*atomiclifetime
peak=gamma/50.
OD=N_z*sigma_0*max_z*1e8
rho_0=basis(2,0)*basis(2,0).dag()
rabi_t=(1 + 1j)*np.zeros([len(tlist),len(zlist)])
rabi_t[:,0]=input_pulse(tlist,t0,peak,width)
excited_pop=np.zeros([len(tlist),len(zlist)])
print(OD)


for i in range(len(zlist)-1):
	time_evolution=solve_Lindblad(tlist,rabi_t[:,i],rho_0)
	excited_pop[:,i]=time_evolution.expect[0]
	rho_eg=time_evolution.expect[1]
	rabi_t[:,i+1]=solve_spatial(rho_eg,N_z,rabi_t[:,i],dz,g)

plt.clf()
plt.contourf(zlist,tlist,abs(rabi_t)**2, 20,alpha = 1, cmap = 'cividis')
plt.xlabel(r"Distance [$\mu$m]")
plt.ylabel(r"Time [ns]")
cbar=plt.colorbar()
cbar.set_label(r"Rabi frequency squared")
plt.title(r"$|\Omega|^2$ vs time and space")
plt.savefig('rabi.pdf')
plt.clf()
plt.contourf(zlist,tlist,excited_pop, 20,alpha = 1, cmap = 'Greens')
plt.xlabel(r"Distance [$\mu$m]")
plt.ylabel(r"Time [ns]")
cbar=plt.colorbar()
cbar.set_label(r"Excited state population")
plt.title(r"$\rho_{ee}$ vs time and space")
plt.savefig('rho_ee.pdf')

# plt.legend(('ground', 'pulse', 'field'),loc='upper right')
# plt.show()

# np.savetxt('populations_2level.dat', data.T)