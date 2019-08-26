#!/usr/bin/env python

import numpy as np
import curie_weiss_periodic as cp
from numpy.linalg import eigh
from scipy.sparse import dia_matrix
import matplotlib.pyplot as plt

#At Freezing Resonance
#besj_root = 2.40482555769577

#Not at Freezing Resonance
besj_root = 1.5

fsize = 16

def get_jmat_pbc(lsize, beta):
    N = lsize
    J = dia_matrix((N, N))
    mid_diag = np.floor(N/2).astype(int)
    for i in xrange(1,mid_diag+1):
        elem = pow(i, -beta)
        J.setdiag(elem, k=i)
        J.setdiag(elem, k=-i)
    for i in xrange(mid_diag+1, N):
        elem = pow(N-i, -beta)
        J.setdiag(elem, k=i)
        J.setdiag(elem, k=-i)
    return J.toarray()

N = 4
t = np.linspace(0.0,200.0,10000)
hz = 3.0
betas = np.linspace(10.,0.0,100)
betas_plot = [8,4,2,0]

dd_avg = []

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(2, 1)
#fig.suptitle(r"At Nearest-Neighbour Freezing with $N=4, J_0\left(\frac{2hN}{\omega}\right)=0$")
fig.suptitle(r"Away from Nearest-Neighbour Freezing with $N=4, J_0\left(\frac{2hN}{\omega}\right)\neq 0$")
axs[0].set_xlabel(r"$t$",fontsize=fsize)
axs[0].set_ylabel(r"$\rho(t)$",fontsize=fsize)

for b in betas:
    print "running for beta=", b
    #Long Range hopping matrix
    J = get_jmat_pbc(N, b)
    #Kac norm
    mid = np.floor(N/2).astype(int)
    kac_norm = 2.0 * np.sum(1/(pow(np.arange(1, mid+1), b).astype(float)))
    #resonance criteria
    f = 2.0 * hz * kac_norm/besj_root
    p = cp.ParamData(hopmat = -J/kac_norm, lattice_size=N, omega=f, times=t, hz=hz, jx=-1.0, jz=0.0)
    h = cp.Hamiltonian(p)
    E, U = eigh(h.hamiltmat + h.trans_hamilt)
    out = cp.run_dyn(p, initstate=U[0]) #Initial condition is t=0 Hamiltonian ground state
    dd_avg.append(np.average(out["defect_density"]))

for b in betas_plot:
    print "plotting for beta=", b
    #Long Range hopping matrix
    J = get_jmat_pbc(N, b)
    #Kac norm
    mid = np.floor(N/2).astype(int)
    kac_norm = 2.0 * np.sum(1/(pow(np.arange(1, mid+1), b).astype(float)))
    #resonance criteria
    f = 2.0 * hz * kac_norm/besj_root
    p = cp.ParamData(hopmat = -J/kac_norm, lattice_size=N, omega=f, times=t, hz=hz, jx=-1.0, jz=0.0)
    h = cp.Hamiltonian(p)
    E, U = eigh(h.hamiltmat + h.trans_hamilt)
    out = cp.run_dyn(p, initstate=U[0]) #Initial condition is t=0 Hamiltonian ground state
    axs[0].plot(out["t"], out["defect_density"], label=r"$\beta=%i$"%b)

axs[0].legend()

#filename="resonance_dd_betas.dat"
filename="off_resonance_dd_betas.dat"
np.savetxt(filename, np.vstack((betas,dd_avg)).T)

axs[1].set_xlabel(r"\beta", fontsize=fsize)
axs[1].set_ylabel(r"$\rho_{avg}$", fontsize=fsize)
axs[1].plot(betas,dd_avg,"--")
#plt.savefig("resonance_dd_betas.png")
plt.savefig("off_resonance_dd_betas.png")
plt.show()
