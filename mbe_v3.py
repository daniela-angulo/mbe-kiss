"""
=====================================================================
=====================================================================

Written by: Daniela Angulo

Program Name: mbe_v2.py

Description: solves the Maxwell-Bloch equations for an electric field with gaussian shape (rabi frequency)

Extra Info: Based on qutip examples and the thesis by Thomas P. Ogden

=====================================================================
=====================================================================
"""

import numpy as np
from matplotlib import rc
import matplotlib as mpl 
from matplotlib import pylab as plt
from numpy import linalg
from scipy.special import factorial
import time as time
from scipy import special
from qutip import * 

starttime = time.time()

mpl.rcParams.update({'font.size': 14,'text.usetex':True})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

#constants 
Isat = 3.9e1#Saturation intensity, F=3->F'=4, (isotropic light polarization) in W/m^2
lambda0 = 780e-9#vacuum wavelengthm
c = 3e8 #m/s
omega0 = 2*np.pi*c/lambda0
hbar = 1.0545718e-34
gamma = 2*np.pi*6.07e-3 #atomic lifetime in GHz
atomiclifetime = 1/gamma #atomic lifetime in ns
a0 = .52917720859e-10 #Bohr radius in meters
e = 1.602176487e-19 #charge of an electron in Coulombs
matrix_element = 1.956*e*a0 #Effective Dipole Moment F=3->F'=4 (isotropic light polarization) in Coulomb meters
epsilon0 = 8.854187817e-12 #permittivity of vacuum in farads/meter
g = (2*np.pi/lambda0)*matrix_element**2/(epsilon0*hbar) #light matter coupling constant in Hz and m^2
sigma_0=1.246e-9 #resonant cross section cm^2

#gaussian envelope
def input_pulse(t,t0,peak,sigma):
	gaussian=peak*np.exp(-(t-t0)**2/4/sigma**2)
	return gaussian

def photon_number(rabi_0,waist):
	energy=2*np.pi*waist**2*Isat*np.sum(abs(rabi_0)**2)*dt*1e-9/gamma**2
	num_photons=energy/(hbar*omega0)
	return num_photons

#solver of the Lindblad equation using qutip, the rabi frequency is time dependent
def solve_Lindblad(t,rabi,rho_0):
	S_12=tensor(basis(2,0)*basis(2,1).dag())
	S_11=tensor(basis(2,0)*basis(2,0).dag())
	S_22=tensor(basis(2,1)*basis(2,1).dag())
	H0=det*S_22
	H1=S_12+S_12.dag()
	H=[H0,[H1,rabi/2.]]
	result = mesolve(H,rho_0,t,np.sqrt(gamma)*S_12,[S_22,S_12.dag()])
	return result

#one step of spatial evolution or paraxial equation
def solve_spatial(rho_z,N_z,rabi_0,dz,g):
	rabi_1=rabi_0+1j*g*(1e3)*dz*rho_z*N_z
	return rabi_1

dt=1.
t_min=-40.
t_max=150.
# dt=30.
# t_min=-1500.
# t_max=1500.
tlist=np.arange(t_min,t_max,dt)
dz=5.
min_z=0
max_z=600.
zlist=np.arange(min_z,max_z,dz)
N=8.e-2 #atoms/um^3
N_z=N*np.pi*25**2 #atoms/um
t0=0.
det=0
# width=0.3817*atomiclifetime
width=10.
peak=gamma*0.00815
# peak=gamma*0.00116
OD=N*sigma_0*max_z*1e8
psi_0=basis(2,0)
rho_0=basis(2,0)*basis(2,0).dag()
rabi_t=(1 + 1j)*np.zeros([len(tlist),len(zlist)])
rabi_t[:,0]=input_pulse(tlist,t0,peak,width)
excited_pop=np.zeros([len(tlist),len(zlist)])
photons=photon_number(rabi_t[:,0],25e-6)

print(OD,photons)


for i in range(len(zlist)-1):
	time_evolution=solve_Lindblad(tlist,rabi_t[:,i],rho_0)
	excited_pop[:,i]=time_evolution.expect[0]
	rho_eg=time_evolution.expect[1]
	rabi_t[:,i+1]=solve_spatial(rho_eg,N,rabi_t[:,i],dz,g)

intensity_isat=2*np.real(rabi_t*np.conj(rabi_t))/gamma**2
time_int_excpop=np.sum(excited_pop,axis=0)*dt
time_int_intensity=np.sum(intensity_isat,axis=0)*dt

#Rate equations
Ne_t_z=N_z*excited_pop
R_sp=Ne_t_z*gamma
dNdt=np.gradient(Ne_t_z,axis=0)/dt
R_diff=R_sp+dNdt

#Find R_abs and R_em
low_diff=R_diff<0
high_diff=R_diff>0
R_abs=np.array(R_diff)
R_em=np.array(R_diff)
R_abs[low_diff]=0
R_em[high_diff]=0

#Integrate the rates to get number of excited atoms per z
L_diff=np.sum(R_diff,axis=0)*dt
L_sp=np.sum(R_sp,axis=0)*dt
L_abs=np.sum(R_abs,axis=0)*dt
L_em=np.sum(abs(R_em),axis=0)*dt

N_diff=np.sum(L_diff)*dz
N_abs=np.sum(L_abs)*dz
N_em=np.sum(L_em)*dz
N_sp=np.sum(L_sp)*dz


#Average times
T_abs=np.sum((R_abs.T*tlist).T,axis=0)[:-1]*dt/L_abs[:-1]
T_em=np.sum((abs(R_em).T*tlist).T,axis=0)[:-1]*dt/L_em[:-1]
T_f=T_em-T_abs

distance=np.int((200-min_z)/dz)

plt.clf()
plt.contourf(zlist,tlist,abs(rabi_t)**2, 40,alpha = 1, cmap = 'cividis')
plt.xlabel(r"Distance [$\mu$m]")
plt.ylabel(r"Time [ns]")
cbar=plt.colorbar()
cbar.set_label(r"Rabi frequency squared")
plt.title(r"$|\Omega|^2$ vs time and space")
plt.savefig('rabi.pdf')
plt.clf()
plt.contourf(zlist,tlist,excited_pop, 40,alpha = 1, cmap = 'cividis')
plt.xlabel(r"Distance [$\mu$m]")
plt.ylabel(r"Time [ns]")
cbar=plt.colorbar()
cbar.set_label(r"Excited state population")
plt.title(r"$\rho_{ee}$ vs time and space")
plt.savefig('rho_ee.pdf')

plt.clf()
plt.ticklabel_format(axis='y',style='sci',scilimits=(-4,0))
plt.plot(tlist,intensity_isat[:,np.int((0-min_z)/dz)],tlist,intensity_isat[:,np.int((250-min_z)/dz)],tlist,intensity_isat[:,np.int((500-min_z)/dz)],'-')
plt.xlabel(r"Time [ns]")
plt.title(r"Intensity at slices")
plt.ylabel(r"Intensity")
plt.show()


plt.clf()
plt.ticklabel_format(axis='y',style='sci',scilimits=(-4,0))
plt.plot(tlist,excited_pop[:,np.int((0-min_z)/dz)],tlist,excited_pop[:,np.int((250-min_z)/dz)],tlist,excited_pop[:,np.int((500-min_z)/dz)],'-')
plt.xlabel(r"Time [ns]")
plt.title(r"Excited state pop at slices")
plt.ylabel(r"$\rho_{ee}$")
plt.show()


plt.clf()
plt.plot(zlist[:-1],time_int_excpop[:-1],'D',markevery=2)
plt.xlabel(r"Distance [$\mu$m]")
plt.ylabel(r"Integral")
plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,0))
plt.show()

plt.clf()
plt.ticklabel_format(axis='y',style='sci',scilimits=(-4,0))
plt.plot(tlist,R_sp[:,distance]*dt*dz,tlist,dNdt[:,distance]*dt*dz,tlist,R_diff[:,distance]*dt*dz,tlist,R_abs[:,distance]*dt*dz,tlist,abs(R_em[:,distance])*dt*dz,'-')
plt.xlabel(r"Time [ns]")
# plt.xlabel(r"Distance [$\mu$m]")
plt.title(r"Rates as a function of time at $z=0\mu$m ")
plt.ylabel(r"Excited atoms per unit length per second")
plt.legend((r'$N_e \Gamma$', r'$dN_e/dt$',r'$R_{diff}$',r'$R_{abs}$',r'$R_{em}$'),loc='upper right')
plt.savefig('rates.pdf')
plt.show()

plt.clf()
plt.plot(zlist[:-1],L_diff[:-1]*dz,'D',zlist[:-1],L_abs[:-1]*dz,'D',zlist[:-1],L_em[:-1]*dz,'D',markevery=5)
plt.xlabel(r"Distance [$\mu$m]")
plt.ylabel(r"Integral in time")
plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,0))
plt.legend((r'$L_{diff}$',r'$L_{abs}$',r'$L_{em}$'),loc='upper right')
plt.savefig('numerperlength.pdf')
stoptime = time.time()
print("Program took %1.2f seconds" %(stoptime-starttime))
plt.show()
