#!/usr/bin/env python
"""
Created on Nov 14 2018

@author: Analabha Roy (daneel@utexas.edu)
"""
import numpy as np
from pprint import pprint
from itertools import combinations
from scipy.integrate import odeint
from tempfile import mkdtemp
import os.path as path

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
    def __init__(self, hopmat = np.eye(11), lattice_size=11, omega=0.0, \
                                      times = np.linspace(0.0, 10.0,100),\
                                       hx=0.0, hy=0.0, hz=0.0,\
                                        jx=0.0, jy=0.0, jz=1.0, memmap=False,\
                                                                verbose=False):
        self.lattice_size = lattice_size
        self.omega = omega
        self.times = times
        self.jx, self.jy, self.jz = jx, jy, jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.verbose = verbose
        self.jmat = hopmat
        self.memmap = memmap
        self.norm = 1.0 #Change to kac norm later

class Hamiltonian:
    description = """Precalculates all the dynamics information
                     of the Hamiltonian"""

    def nummats(self, mu, memmap=False):
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
        if memmap:
            fname_x = path.join(mkdtemp(), 'ntemp_x.dat')
            fname_y = path.join(mkdtemp(), 'ntemp_y.dat')
            fname_z = path.join(mkdtemp(), 'ntemp_z.dat')
            fp_x = np.memmap(fname_x, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_x[:,:] = num_x[:,:]
            fp_y = np.memmap(fname_y, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_y[:,:] = num_y[:,:]
            fp_z = np.memmap(fname_z, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_z[:,:] = num_z[:,:]
            return (fp_x, fp_y, fp_z)
        else:
            return (num_x, num_y, num_z)

    def kemats(self, sitepair, memmap=False):
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
        if memmap:
            fname_x = path.join(mkdtemp(), 'ktemp_x.dat')
            fname_y = path.join(mkdtemp(), 'ktemp_y.dat')
            fname_z = path.join(mkdtemp(), 'ktemp_z.dat')
            fp_x = np.memmap(fname_x, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_x[:,:] = ke_x[:,:]
            fp_y = np.memmap(fname_y, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_y[:,:] = ke_y[:,:]
            fp_z = np.memmap(fname_z, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_z[:,:] = ke_z[:,:]
            return (fp_x, fp_y, fp_z)
        else:
            return (ke_x, ke_y, ke_z)

    def offd_corrmats(self, sitepair, memmap=False):
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
        if memmap:
            fname_x = path.join(mkdtemp(), 'ctemp_x.dat')
            fname_y = path.join(mkdtemp(), 'ctemp_y.dat')
            fname_z = path.join(mkdtemp(), 'ctemp_z.dat')
            fp_x = np.memmap(fname_x, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_x[:,:] = cxy[:,:]
            fp_y = np.memmap(fname_y, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_y[:,:] = cxz[:,:]
            fp_z = np.memmap(fname_z, dtype='complex128', mode='w+', shape=(2**lsize,2**lsize))
            fp_z[:,:] = cyz[:,:]
            return (fp_x, fp_y, fp_z)
        else:
            return (cxy, cxz, cyz)

    def __init__(self, params):
        #Copy arguments from params to this class
        self.__dict__.update(params.__dict__)
        #Build KE matrix
        self.hamiltmat = self.jx * np.sum(np.array(\
          [self.jmat[sitepair] * self.kemats(sitepair, memmap=params.memmap)[0] \
            for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
        self.hamiltmat += self.jy * np.sum(np.array(\
          [self.jmat[sitepair] * self.kemats(sitepair, memmap=params.memmap)[1] \
            for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
        self.hamiltmat += self.jz * np.sum(np.array(\
          [self.jmat[sitepair] * self.kemats(sitepair, memmap=params.memmap)[2] \
            for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
        self.hamiltmat = self.hamiltmat/self.norm
        self.trans_hamilt = self.hx * np.sum(np.array([self.nummats(mu, memmap=params.memmap)[0] \
                                             for mu in xrange(self.lattice_size)]), axis=0)
        self.trans_hamilt += self.hy * np.sum(np.array([self.nummats(mu, memmap=params.memmap)[1] \
                                              for mu in xrange(self.lattice_size)]), axis=0)
        self.trans_hamilt += self.hz * np.sum(np.array([self.nummats(mu, memmap=params.memmap)[2] \
                                              for mu in xrange(self.lattice_size)]), axis=0)

def jac(y, t0, jacmat, hamilt, params):
    omega = params.omega
    drive = np.cos(omega*t0)
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


def defect_density(hamilt):
    """
    Returns a matrix (sum_k \gamma^\dagger_k \gamma_k) that asymptotically
    approaches the Defect Density Operator in the thermodynamic limit
    """
    lsize = hamilt.lattice_size
    fbz = np.linspace(-np.pi, np.pi, lsize)
    epsilon_k = hamilt.hz + np.cos(fbz)
    delta_k = np.sin(fbz)
    E_k =  np.sqrt(epsilon_k * epsilon_k + delta_k * delta_k)
    s = delta_k/(epsilon_k - E_k)
    u = np.sqrt(1/(1 + s * s))
    v = s * u
    u[np.isnan(v)] = 0.0
    v[np.isnan(v)] = 1.0
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
        ck_ops.append(np.sum([c_j * np.exp((1j)*k*j)/np.sqrt(lsize) for j, c_j in enumerate(c_ops)], axis=0))

    gammak_ops = []
    for ki, k in enumerate(fbz):
        if k > 0:
            gammak_ops.append(u[ki] * ck_ops[ki] - (1j) * v[ki] * np.conjugate(ck_ops[::-1][ki].T))
    return np.sum([np.dot(g.T.conjugate(), g) for g in gammak_ops], axis=0)/lsize

def evolve_numint(hamilt,times,initstate, params):
    (rows,cols) = hamilt.hamiltmat.shape
    fulljac = np.zeros((2*rows,2*cols), dtype="float64")
    fulljac[0:rows, 0:cols] = hamilt.hamiltmat.imag
    fulljac[0:rows, cols:] = hamilt.hamiltmat.real
    fulljac[rows:, 0:cols] = -hamilt.hamiltmat.real
    fulljac[rows:, cols:] = hamilt.hamiltmat.imag

    psi_t = odeint(func, np.concatenate((initstate.real, initstate.imag)),\
      times, args=(fulljac, hamilt, params), Dfun=jac)
    return psi_t[:,0:rows] + (1.j) * psi_t[:, rows:]


def run_dyn(params, initstate=None):
    if params.verbose:
        print "Executing diagonalization with parameters:"
        pprint(vars(params), depth=1)

    h = Hamiltonian(params)
    lsize = h.lattice_size

    #Assume that psi_0 is the eigenstate of \sum_\mu sigma^x_\mu
    if initstate is None:
        initstate =  np.ones(2**lsize, dtype="float64")/np.sqrt(2**lsize)
    #Start from ground state
    #E, U = LA.eigh(h.hamiltmat + h.trans_hamilt)
    #initstate = U[0]

    #Required observables

    sx = np.sum(np.array([h.nummats(mu)[0] for mu in xrange(lsize)]), axis=0)
    sy = np.sum(np.array([h.nummats(mu)[1] for mu in xrange(lsize)]), axis=0)
    sz = np.sum(np.array([h.nummats(mu)[2] for mu in xrange(lsize)]), axis=0)
    sx, sy, sz = sx/lsize, sy/lsize, sz/lsize

    #Defect density
    gdg = defect_density(h)

    #SZSZ Correlations
    mid = np.floor(lsize/2).astype(int)
    sites = np.concatenate((np.arange(mid),np.arange(mid+1, lsize)))
    
    szsz =  np.array([np.array(h.kemats((np.minimum(i,mid),np.maximum(i,mid)),\
                                        memmap=params.memmap)[2]) for i in sites])

    psi_t = evolve_numint(h, params.times, initstate, params)

    sxdata = np.array([np.vdot(psi,np.dot(sx,psi)) for psi in psi_t])
    sydata = np.array([np.vdot(psi,np.dot(sy,psi)) for psi in psi_t])
    szdata = np.array([np.vdot(psi,np.dot(sz,psi)) for psi in psi_t])

    defect_density_data = np.abs(np.array([np.vdot(psi,np.dot(gdg,psi))\
                                           for psi in psi_t]))
            
    szszdata = np.real(np.array([np.dot(np.conj(psi),np.dot(szsz,psi).T) for psi in psi_t]))

    print "\nDumping outputs to dictionary ..."

    return {"t":params.times, "sx":sxdata, "sy":sydata, "sz":szdata, \
                                          "defect_density":defect_density_data,\
                                          "szsz":szszdata}

if __name__ == '__main__':
    #Power law decay of interactions
    paramdat = ParamData()
    run_dyn(paramdat)
