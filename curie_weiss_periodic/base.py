#!/usr/bin/env python
"""
Created on Nov 14 2018

@author: Analabha Roy (daneel@utexas.edu)
"""
import numpy as np
from pprint import pprint
from itertools import combinations
from scipy.integrate import odeint

desc = """Dynamics by exact diagonalization of
                generalized Curie-Weiss model with long range interactions
                and periodic drive"""

#Pauli matrices
sig_x, sig_y, sig_z = \
  np.array([[0j, 1.0+0j], [1.0+0j, 0j]]), \
    np.array([[0j, -1j], [1j, 0j]]), \
      np.array([[1.0+0j, 0j], [0j, -1+0j]])
sig_plus = (sig_x + sig_y*1j)/2.0
sig_minus = (sig_x - sig_y*1j)/2.0

class ParamData:
    description = """Class to store parameters and hopping matrix"""
    def __init__(self, hopmat = np.eye(11), lattice_size=11, \
                                      ampl=1.0, omega=0.0,\
                                      times = np.linspace(0.0, 10.0,100),\
                                       hx=0.0, hy=0.0, hz=0.0,\
                                        jx=0.0, jy=0.0, jz=1.0, verbose=False):
        self.lattice_size = lattice_size
        self.ampl = ampl
        self.omega = omega
        self.times = times
        self.jx, self.jy, self.jz = jx, jy, jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.verbose = verbose
        self.jmat = hopmat
        self.norm = 1.0 #Change to kac norm later

class Hamiltonian:
    description = """Precalculates all the dynamics information
                     of the Hamiltonian"""

    def nummats(self, mu):
        lsize = self.lattice_size
        #Left Hand Side
        if(mu == 0):
            num_x, num_y, num_z  = sig_x, sig_y, sig_z
        else:
            id = np.eye(2**mu,2**mu)
            num_x, num_y, num_z  = \
              np.kron(id,sig_x), np.kron(id,sig_y), np.kron(id,sig_z)
        #Right Hand Side
        if(mu < lsize - 1):
            id = np.eye(2**(lsize-mu-1),2**(lsize-mu-1))
            num_x, num_y, num_z = \
              np.kron(num_x, id), np.kron(num_y, id), np.kron(num_z, id)
        return (num_x, num_y, num_z)

    def kemats(self, sitepair):
        lsize = self.lattice_size
        (mu, nu) = sitepair
        #Left Hand Side
        if(mu == 0):
            ke_x, ke_y, ke_z = sig_x, sig_y, sig_z
        else:
            id = np.eye(2**mu,2**mu)
            ke_x, ke_y, ke_z = \
              np.kron(id, sig_x), np.kron(id, sig_y), np.kron(id, sig_z)
        #Middle Side
        dim = 1 if mu == nu else 2**(np.abs(mu-nu)-1)
        id = np.eye(dim,dim)
        ke_x, ke_y, ke_z = \
          np.kron(ke_x, id), np.kron(ke_y, id), np.kron(ke_z, id)
        ke_x, ke_y, ke_z = \
          np.kron(ke_x, sig_x), np.kron(ke_y, sig_y), np.kron(ke_z, sig_z)
        #Right Hand Side
        if(nu < lsize - 1):
            id = np.eye(2**(lsize-nu-1),2**(lsize-nu-1))
            ke_x, ke_y, ke_z = \
            np.kron(ke_x, id), np.kron(ke_y, id), np.kron(ke_z, id)
        return (ke_x, ke_y, ke_z)

    def offd_corrmats(self, sitepair):
        lsize = self.lattice_size
        (mu, nu) = sitepair
        #Left Hand Side
        if(mu == 0):
            cxy, cxz, cyz = sig_x, sig_x, sig_y
        else:
            id = np.eye(2**mu,2**mu)
            cxy, cxz, cyz = \
              np.kron(id, sig_x), np.kron(id, sig_x), np.kron(id, sig_y)
        #Middle Side
        dim = 1 if mu == nu else 2**(np.abs(mu-nu)-1)
        id = np.eye(dim,dim)
        cxy, cxz, cyz = \
          np.kron(cxy, id), np.kron(cxz, id), np.kron(cyz, id)
        cxy, cxz, cyz = \
          np.kron(cxy, sig_y), np.kron(cxz, sig_z), np.kron(cyz, sig_z)
        #Right Hand Side
        if(nu < self.lattice_size - 1):
            id = np.eye(2**(lsize-nu-1),2**(lsize-nu-1))
            cxy, cxz, cyz = \
            np.kron(cxy, id), np.kron(cxz, id), np.kron(cyz, id)
        return (cxy, cxz, cyz)

    def __init__(self, params):
        #Copy arguments from params to this class
        self.__dict__.update(params.__dict__)
        #Build KE matrix
        self.hamiltmat = self.jx * np.sum(np.array(\
          [self.jmat[sitepair] * self.kemats(sitepair)[0] \
            for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
        self.hamiltmat += self.jy * np.sum(np.array(\
          [self.jmat[sitepair] * self.kemats(sitepair)[1] \
            for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
        self.hamiltmat += self.jz * np.sum(np.array(\
          [self.jmat[sitepair] * self.kemats(sitepair)[2] \
            for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
        self.hamiltmat = self.hamiltmat/self.norm
        self.trans_hamilt = self.hx * np.sum(np.array([self.nummats(mu)[0] \
                                             for mu in xrange(self.lattice_size)]), axis=0)
        self.trans_hamilt += self.hy * np.sum(np.array([self.nummats(mu)[1] \
                                              for mu in xrange(self.lattice_size)]), axis=0)
        self.trans_hamilt += self.hz * np.sum(np.array([self.nummats(mu)[2] \
                                              for mu in xrange(self.lattice_size)]), axis=0)

def jac(y, t0, jacmat, hamilt, params):
    omega = params.omega
    ampl = params.ampl
    drive = ampl * np.cos(omega*t0)
    (rows,cols) = hamilt.hamiltmat.shape
    jacmat[0:rows, 0:cols] = hamilt.hamiltmat.imag + \
                                 drive * hamilt.trans_hamilt.imag
    jacmat[0:rows, cols:] = hamilt.hamiltmat.real+ \
                                drive * hamilt.trans_hamilt.real
    jacmat[rows:, 0:cols] = - hamilt.hamiltmat.real - \
                                drive * hamilt.trans_hamilt.real
    jacmat[rows:, cols:] = hamilt.hamiltmat.imag + \
                                drive * hamilt.trans_hamilt.imag
    return jacmat

def func(y, t0, jacmat, hamilt, params):
    return np.dot(jac(y, t0, jacmat, hamilt, params), y)

# =============================================================================
# def defect_density(hamilt):
#     lsize = hamilt.lattice_size
#     kprod_sigZ = np.kron(sig_z,np.eye(2**(lsize-2),2**(lsize-2)))
#     for x in xrange(1,lsize-1):
#         id1 = np.kron(np.eye(2**x,2**x),sig_z)
#         id2 = np.kron(id1,np.eye(2**(lsize-2-x),2**(lsize-2-x)))
#         kprod_sigZ = kprod_sigZ + id2    
#     sigZ_plus = np.kron(kprod_sigZ,sig_plus)
#     sigZ_minus = np.kron(kprod_sigZ,sig_minus)
#     bglbv_d = np.zeros((2**lsize,2**lsize), dtype='complex128')
#     # run the loops
#     for i in xrange(0,lsize):
#         k = -np.pi
#         while k<=np.pi:
#             h0 = 10
#             eps_k = h0 + np.cos(k)
#             e_k = np.sqrt(np.sin(k)**2 np.eye(2**lsize)+ (h0 + np.cos(k)**2))
#             delt_k = np.sin(k)
#     
#             mm = (e_k - eps_k)/delt_k
#             mm1 = (1-0.5*(mm**2))
#             
#             expn_minus = ((np.exp(-lsize*k*1j)-1)/(np.exp(-k*1j-1)))**2
#             expn_plus = ((np.exp(lsize*k*1j)-1)/(np.exp(k*1j-1)))**2
#             # gamma dagger
#             gammad = (mm1*expn_minus*sigZ_plus)*1j + mm * mm1 * expn_minus * sigZ_minus
#             gamma = mm * mm1 * expn_plus * sigZ_plus - (mm1 * expn_plus * sigZ_minus)*1j 
#             bglbv_d += np.dot(gammad,gamma)
#             k+=0.01
#     return bglbv_d        
# =============================================================================


#Problem with normalization!!!!
def defect_density(hamilt):
    lsize = hamilt.lattice_size
    fbz = np.linspace(-np.pi, np.pi, lsize)
    epsilon_k = hamilt.ampl + np.cos(fbz)
    delta_k = np.sin(fbz)
    E_k = - np.sqrt(epsilon_k * epsilon_k + delta_k * delta_k)
    s = (epsilon_k - E_k)/delta_k
    v = 1.0 - 0.5 * s * s
    u = s * v
    c_ops = []
    for i in np.arange(lsize):
        if i != 0:
            c_i = 1
            for j in np.arange(0,i):
                c_i = np.kron(c_i, sig_z)
            c_i = np.kron(c_i, sig_plus)    
        else:
            c_i = sig_plus
        c_i = np.kron(c_i,np.eye(2**(lsize - i - 1)))    
        c_ops.append(c_i)
    ck_ops = []
    for k in fbz:
        ck_ops.append(np.sum([c_j * np.exp((1j)*k*j) for j, c_j in enumerate(c_ops)], axis=0)/np.sqrt(lsize))
    gammak_ops = []
    for ki, k in enumerate(fbz):
        gammak_ops.append(u[ki] * ck_ops[ki] - (1j) * v[ki] * ck_ops[::-1][ki])
    return np.sum([np.dot(gammak_ops[ki].T.conjugate(), gammak_ops[ki]) for ki, k in enumerate(fbz)], axis=0)/lsize    


def evolve_numint(hamilt,times,initstate, params):
    (rows,cols) = hamilt.hamiltmat.shape
    fulljac = np.zeros((2*rows,2*cols), dtype="float64")
    fulljac[0:rows, 0:cols] = hamilt.hamiltmat.imag
    fulljac[0:rows, cols:] = hamilt.hamiltmat.real
    fulljac[rows:, 0:cols] = -hamilt.hamiltmat.real
    fulljac[rows:, cols:] = hamilt.hamiltmat.imag

    psi_t = odeint(func, np.concatenate((initstate, np.zeros(rows))),\
      times, args=(fulljac, hamilt, params), Dfun=jac)
    return psi_t[:,0:rows] + (1.j) * psi_t[:, rows:]


def run_dyn(params):
    if params.verbose:
        print "Executing diagonalization with parameters:"
        pprint(vars(params), depth=1)
    else:
        print "Starting run ..."

    h = Hamiltonian(params)
    lsize = h.lattice_size
    lsq = lsize * lsize

    #Assume that psi_0 is the eigenstate of \sum_\mu sigma^x_\mu
    initstate =  np.ones(2**lsize, dtype="float64")/np.sqrt(2**lsize)
    t_output = params.times

    #Required observables

    sx = np.sum(np.array([h.nummats(mu)[0] \
      for mu in xrange(h.lattice_size)]), axis=0)
    sy = np.sum(np.array([h.nummats(mu)[1] \
      for mu in xrange(h.lattice_size)]), axis=0)
    sz = np.sum(np.array([h.nummats(mu)[2] \
      for mu in xrange(h.lattice_size)]), axis=0)
    sx, sy, sz = sx/lsize, sy/lsize, sz/lsize

    sxvar = np.sum(np.array( [h.kemats(sitepair)[0] \
          for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    syvar = np.sum(np.array( [h.kemats(sitepair)[1] \
      for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    szvar = np.sum(np.array( [h.kemats(sitepair)[2] \
      for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    sxvar, syvar, szvar = (sxvar/lsq), (syvar/lsq), (szvar/lsq)

    sxyvar = np.sum(np.array(\
      [ h.offd_corrmats(sitepair)[0] \
        for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    sxzvar = np.sum(np.array(\
      [ h.offd_corrmats(sitepair)[1] \
        for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    syzvar = np.sum(np.array(\
      [h.offd_corrmats(sitepair)[2] \
        for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    sxyvar, sxzvar, syzvar = (sxyvar/lsq), (sxzvar/lsq), (syzvar/lsq)
    
    #Defect density
    gdg = defect_density(h)

    psi_t = evolve_numint(h, t_output, initstate, params)

    sxdata = np.array([np.vdot(psi,np.dot(sx,psi)) for psi in psi_t])
    sydata = np.array([np.vdot(psi,np.dot(sy,psi)) for psi in psi_t])
    szdata = np.array([np.vdot(psi,np.dot(sz,psi)) for psi in psi_t])

    sxvar_data = np.array([np.vdot(psi,np.dot(sxvar,psi)) \
      for psi in psi_t])
    sxvar_data = 2.0 * sxvar_data + (1./lsize) - (sxdata)**2

    syvar_data = np.array([np.vdot(psi,np.dot(syvar,psi)) \
      for psi in psi_t])
    syvar_data = 2.0 * syvar_data + (1./lsize) - (sydata)**2

    szvar_data = np.array([np.vdot(psi,np.dot(szvar,psi)) \
      for psi in psi_t])
    szvar_data = 2.0 * szvar_data + (1./lsize) - (szdata)**2

    sxyvar_data = np.array([np.vdot(psi,np.dot(sxyvar,psi)) \
      for psi in psi_t])
    sxyvar_data = 2.0 * sxyvar_data - (sxdata) * (sydata)

    sxzvar_data = np.array([np.vdot(psi,np.dot(sxzvar,psi)) \
      for psi in psi_t])
    sxzvar_data = 2.0 * sxzvar_data - (sxdata) * (szdata)

    syzvar_data = np.array([np.vdot(psi,np.dot(syzvar,psi)) \
      for psi in psi_t])
    syzvar_data = 2.0 * syzvar_data - (sydata) * (szdata)
    
    defect_density_data = np.array([np.vdot(psi,np.dot(gdg,psi)) for psi in psi_t])

    print "\nDumping outputs to files ..."
    print np.vstack(zip(t_output,sxdata))

    #DO THIS
    
    print np.vstack(zip(t_output,defect_density_data))
    print 'Done'

if __name__ == '__main__':   
    #Power law decay of interactions
    paramdat = ParamData()
    run_dyn(paramdat)
