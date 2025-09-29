import numpy as np
from qutip import *
import os
from matplotlib import pyplot as plt
import sys
import pickle
from scipy.integrate import solve_ivp

def lindblad_term(rho, a):
    return a@rho@np.conj(a.T) - .5*(np.conj(a.T)@a@rho + rho@np.conj(a.T)@a)

def commutator(A, B):
    return A@B - B@A

def lindblad_master_equation(t, rho_vec, hamiltonian, dissipation_channels):
    d = int(len(rho_vec)**.5)
    rho = rho_vec.reshape(d, d)
    drho = -1j*commutator(hamiltonian,rho) + np.sum([lindblad_term(rho, i) for i in dissipation_channels], axis=0)
    return drho.flatten()
    # if callable(hamiltonian):
    #     return commutator(hamiltonian(t),rho) + np.sum([lindblad_term(rho, i) for i in dissipation_channels], axis=0)
    # else

def main():
    dyn_name = "example"

    outdir = "output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    N = 20

    cavity_diss_rate = 2*np.pi*4.3*10**6
    input_power = 0
    cavity_photon_number = 10**(input_power/10)
    rabi_freq = 2*np.pi*9*10**6
    eff_coupling = 2*np.pi*(1)*10**6
    qubit_targetz = 9*2*np.pi*10**6
    qubit_detuning = qubit_targetz + eff_coupling*(2*cavity_photon_number + 1) 

    qubit_detuning_lamb_shift = qubit_detuning - eff_coupling*(2*cavity_photon_number + 1)
    cavity_detuning = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5

    cavity_drive_amplitude = np.real((cavity_photon_number * (cavity_detuning**2 + .25*cavity_diss_rate**2))**.5)
    cavity_field = cavity_drive_amplitude/(-cavity_detuning + .5j*cavity_diss_rate)

    norm = 2*np.pi*10**6
    final_time = 2 # in 1/Mhz
    tlist = np.linspace(0, final_time, 10000)/(10**6)

    tlist__N = tlist*norm
    cavity_detuning__N = cavity_detuning/norm
    cavity_drive_amplitude__N = cavity_drive_amplitude/norm
    qubit_detuning__N = qubit_detuning/norm
    eff_coupling__N = eff_coupling/norm
    rabi_freq__N = rabi_freq/norm
    cavity_diss_rate__N = cavity_diss_rate/norm

    d = tensor(destroy(N), qeye(2))   
    sz = tensor(qeye(N), sigmaz())  
    sx = tensor(qeye(N), sigmax())     
    sm = tensor(qeye(N), sigmam()) 

    d_matrix = d.full()
    sz_matrix = sz.full()
    sx_matrix = sx.full()

    H_cav = cavity_detuning__N * d.dag()*d + cavity_drive_amplitude__N * (d + d.dag())
    H_qubit = -.5*(qubit_detuning__N - eff_coupling__N)*sz - .5*rabi_freq__N*sx
    H_int =   eff_coupling__N*d.dag()*d*sz
    H = H_cav + H_qubit + H_int

    H_matrix = H.full()


    dissipation_channels = [(cavity_diss_rate__N**.5*d_matrix)]

    psi0_atom = basis(2,1)
    psi0_cavity = coherent(N,cavity_field)
    initial_state = tensor(psi0_cavity, psi0_atom)
    rho0 = initial_state@initial_state.dag()
    rho0_matrix = rho0.full()
    rho0_vectorized = rho0_matrix.flatten()


    max_step =  0.05/np.max([cavity_drive_amplitude__N, qubit_detuning__N, cavity_detuning__N,
                            rabi_freq__N, eff_coupling__N, cavity_diss_rate__N])

    sol = solve_ivp(lindblad_master_equation, (tlist__N[0], tlist__N[-1]), rho0_vectorized, method="DOP853", t_eval=tlist__N, max_step=max_step, args=(H_matrix, dissipation_channels))

    dynamics = sol.y.T.reshape(10000, 40, 40)


    parameters = {"N":N,
                "final_time": final_time,
                "tlist": tlist,
                "input_power": input_power,
                "cavity_detuning": cavity_detuning,
                "rabi_freq": rabi_freq,
                "cavity_diss_rate": cavity_diss_rate,
                "norm": norm,
                "initial_state": initial_state}
    
    f = open(outdir + f"/{dyn_name}.pckl", 'wb')
    pickle.dump([parameters, dynamics], f)
    f.close()   