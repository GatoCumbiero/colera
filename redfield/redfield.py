import numpy as np
from qutip import *
import os
from matplotlib import pyplot as plt
import sys
from scipy.integrate import solve_ivp


def commutator(A, B):
    return A@B - B@A

def redfield_master_equation(t, rho_vec, hamiltonian, redfield_operators,
                             coupling_operators):
    
    d = int(len(rho_vec)**.5)
    rho = rho_vec.reshape(d, d)
    drho = -1j*commutator(hamiltonian,rho) + \
            np.sum([commutator(redfield_operators[i]@rho, coupling_operators[i]) for i in range(len(redfield_operators))], axis=0)  + \
            np.sum([commutator(coupling_operators[i].conj().T,rho.T@(redfield_operators[i].conj().T)) for i in range(len(redfield_operators))], axis=0)
        #     np.sum([commutator(redfield_operators[i]@rho, coupling_operators[i]) for i in range(len(redfield_operators))], axis=0).conj().T

    return drho.flatten()

def redfield_master_equation(t, rho_vec, hamiltonian, Z,
                             S):
    d = int(len(rho_vec)**.5)
    rho = rho_vec.reshape(d, d)
    drho = -1j*(hamiltonian@rho - rho@hamiltonian) + Z@rho@S + S@rho@Z.conj().T - S@Z@rho - rho@Z.conj().T@S
    return drho.flatten()  

def calculate_dynamics():
    N = 20

    cavity_diss_rate = 2*np.pi*0.8*10**6
    input_power = 0
    cavity_photon_number = 10**(input_power/10)
    rabi_freq = 2*np.pi*4*10**6
    eff_coupling = 2*np.pi*(1)*10**6
    qubit_targetz = 20*2*np.pi*10**6
    qubit_detuning = qubit_targetz + eff_coupling*(2*cavity_photon_number + 1) 

    qubit_detuning_lamb_shift = qubit_detuning - eff_coupling*(2*cavity_photon_number + 1)
    cavity_detuning = -(qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5

    cavity_drive_amplitude = np.real((cavity_photon_number * (cavity_detuning**2 + .25*cavity_diss_rate**2))**.5)
    cavity_field = cavity_drive_amplitude/(-cavity_detuning + .5j*cavity_diss_rate)

    diag_qubit_freq = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
    coupling_sz = eff_coupling*qubit_detuning_lamb_shift/diag_qubit_freq
    coupling_sx = -eff_coupling*rabi_freq/diag_qubit_freq

    norm = 2*np.pi*10**6
    final_time = 10 # in 1/Mhz
    tlist = np.linspace(0, final_time, 10000)/(10**6)

    tlist__N = tlist*norm
    cavity_detuning__N = cavity_detuning/norm
    cavity_drive_amplitude__N = cavity_drive_amplitude/norm
    qubit_detuning__N = qubit_detuning/norm
    eff_coupling__N = eff_coupling/norm
    rabi_freq__N = rabi_freq/norm
    cavity_diss_rate__N = cavity_diss_rate/norm
    diag_qubit_freq__N = diag_qubit_freq/norm
    coupling_sz__N = coupling_sz/norm
    coupling_sx__N = coupling_sx/norm

    sz = sigmaz()
    sx = sigmax()
    sm = sigmap()

    sz_matrix = sz.full()
    sx_matrix = sx.full()
    sm_matrix = sm.full()


    H_qubit_matrix = -.5*diag_qubit_freq__N*sz_matrix


    u1 = (cavity_photon_number*coupling_sz__N/(-1j*cavity_detuning__N + cavity_diss_rate__N/2)*sz_matrix \
        +  cavity_photon_number*coupling_sx__N/(1j*(diag_qubit_freq__N - cavity_detuning__N) + cavity_diss_rate__N/2)*sm_matrix.T \
        +  cavity_photon_number*coupling_sx__N/(1j*(-diag_qubit_freq__N - cavity_detuning__N) + cavity_diss_rate__N/2)*sm_matrix )

    print(sm_matrix)

    s1 = coupling_sz__N*sz_matrix + coupling_sx__N*sx_matrix

    u_op = [u1]
    s_op = [s1]

    psi0_atom = basis(2,1)
    rho0 = psi0_atom@psi0_atom.dag()
    rho0_matrix = rho0.full()
    rho0_vectorized = rho0_matrix.flatten()


    max_step =  0.05/np.max([cavity_drive_amplitude__N, qubit_detuning__N, cavity_detuning__N,
                            rabi_freq__N, eff_coupling__N, cavity_diss_rate__N])

    sol = solve_ivp(redfield_master_equation, (tlist__N[0], tlist__N[-1]), rho0_vectorized, method="DOP853",
                    t_eval=tlist__N, max_step=max_step, args=(H_qubit_matrix, u1, s1))

    dynamics = sol.y.T.reshape(10000, 2, 2)
    
    z = np.trace(dynamics@sz_matrix, axis1=1, axis2=2)
    x = np.trace(dynamics@sx_matrix, axis1=1, axis2=2)

    xmean = np.mean(x[-20:])
    zmean = np.mean(z[-20:])

    #Create the figures.
    fig, axes = plt.subplots(2, 1, figsize=(4,6))

    #fig.suptitle(fig_title)

    axes[0].plot(tlist*10**6, x, color='#000080', label= f"{round(np.real(xmean),2)}")
    #axes[0].set_xlabel('Time (1/Mhz)')
    axes[0].set_ylabel(r'$ <\sigma_x >$')
    axes[0].legend()
    axes[0].set_ylim(-1, 1)

    axes[1].plot(tlist*10**6, z, color='#008000', label= f"{round(np.real(zmean),2)}")
    #axes[1].set_xlabel('Time (1/Mhz)')
    axes[1].set_ylabel(r'$<\sigma_z >$')
    axes[1].legend()
    axes[1].set_ylim(-1, 1)

    fig.savefig("redfield.png")

    plt.close(fig)


if __name__ == "__main__":
    calculate_dynamics()