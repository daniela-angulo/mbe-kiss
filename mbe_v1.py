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

def input_pulse(t,t0,sigma,det):
	gaussian=9.*np.exp(-(t-t0)**2/2/sigma**2)
	return gaussian

def solve_Lindblad(t,rabi,rho_0):
	gamma=6.
	S_12=tensor(basis(2,0)*basis(2,1).dag())
	S_11=tensor(basis(2,0)*basis(2,0).dag())
	S_22=tensor(basis(2,1)*basis(2,1).dag())
	H0=det*S_22
	H1=S_12+S_12.dag()
	H=[H0,[H1,rabi]]
	result = mesolve(H,rho_0,t,np.sqrt(gamma)*S_12,[S_22])
	return result

t_steps=200
tlist=np.linspace(-20.,20.0, t_steps)
t0=0.
sigma=3.
det=0
rho_0=basis(2,0)*basis(2,0).dag()
rabi=input_pulse(tlist,t0,sigma,det)
time_evolution=solve_Lindblad(tlist,rabi,rho_0)
s_1=time_evolution.expect[0]

data=np.array([tlist,s_1])
plt.plot(tlist,s_1)
plt.legend(('ground', 'pulse', 'field'),loc='upper right')
plt.show()
# plt.savefig('2levelsimple.pdf')
# np.savetxt('populations_2level.dat', data.T)