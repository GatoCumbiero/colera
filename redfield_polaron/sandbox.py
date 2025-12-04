import pickle
from datetime import datetime
import numpy as np
from qutip import *
import os
from matplotlib import pyplot as plt
import sys
from scipy.integrate import solve_ivp

def redfield_master_equation(t, rho_vec, hamiltonian, Z_operators, S_operators):
    d = int(len(rho_vec)**.5)
    rho = rho_vec.reshape(d, d)
    
    Z_conj = np.conj(Z_operators)
    
    term1 = np.einsum('nij,jk,nkl->il', Z_operators, rho, S_operators)
    term2 = np.einsum('nij,jk,nlk->il', S_operators, rho, Z_conj)
    term3 = np.einsum('nij,njk,kl->il', S_operators, Z_operators, rho)  
    term4 = np.einsum('ij,nkj,nkl->il', rho, Z_conj, S_operators)     
    
    redfield_part = term1 + term2 - term3 - term4

    coherent_part = -1j*(np.einsum('ij,jk',hamiltonian, rho) - np.einsum('ij,jk', rho, hamiltonian))
    
    drho = coherent_part + redfield_part
    return drho.flatten()

def lindblad_master_equation(t, rho_vec, hamiltonian, L_operators):
    d = int(len(rho_vec)**.5)
    rho = rho_vec.reshape(d, d)

    L_conj = np.conj(L_operators)

    term1 = np.einsum('nij, jk, nlk', L_operators, rho, L_conj)
    term2 = np.einsum('nji, njk, kl', L_conj, L_operators, rho)
    term3 = np.einsum('ij, nkj, nkl', rho, L_conj, L_operators)

    lindblad_part = term1 - .5*(term2 + term3)

    coherent_part = -1j*(np.einsum('ij,jk',hamiltonian, rho) - np.einsum('ij,jk', rho, hamiltonian))

    drho = coherent_part + lindblad_part
    return drho.flatten()

def calculate_full_dynamics(parameters):
    
    #Retrieve needed parameters from the dictionary.
    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_detuning = parameters['qubit_detuning']
    cavity_drive_amplitude = parameters['cavity_drive_amplitude']
    cavity_detuning = parameters['cavity_detuning']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters["time_steps"]
    psi0_atom = parameters['initial_state_qubit']
    psi0_cavity = parameters['initial_state_cavity']
    tlist = np.linspace(0, final_time__mus, time_steps)/(10**6)

    #Norm everything
    tlist__N = tlist*norm
    cavity_detuning__N = cavity_detuning/norm
    cavity_drive_amplitude__N = cavity_drive_amplitude/norm
    qubit_detuning__N = qubit_detuning/norm
    eff_coupling__N = eff_coupling/norm
    rabi_freq__N = rabi_freq/norm
    cavity_diss_rate__N = cavity_diss_rate/norm

    ### HAMILTONIAN 7 ####
    d = tensor(destroy(N), qeye(2))   
    sz = tensor(qeye(N), sigmaz())  
    sx = tensor(qeye(N), sigmax())     
    sm = tensor(qeye(N), sigmam()) 

    H_cav = cavity_detuning__N * d.dag()*d + cavity_drive_amplitude__N * (d + d.dag())
    H_qubit = .5*(qubit_detuning__N + eff_coupling__N)*sz + .5*rabi_freq__N*sx
    H_int =   eff_coupling__N*d.dag()*d*sz
    H = H_cav + H_qubit + H_int

    H_matrix = H.full()
    d_matrix = d.full()


    dissipation_channels = [(cavity_diss_rate__N**.5*d_matrix)]
    initial_state = tensor(psi0_cavity, psi0_atom)
    rho0 = initial_state@initial_state.dag()
    rho0_matrix = rho0.full()
    rho0_vectorized = rho0_matrix.flatten()

    max_step =  0.05/np.max([cavity_drive_amplitude__N, qubit_detuning__N, cavity_detuning__N,
                            rabi_freq__N, eff_coupling__N, cavity_diss_rate__N])

    sol = solve_ivp(lindblad_master_equation, (tlist__N[0], tlist__N[-1]), rho0_vectorized, method="DOP853",
                     t_eval=tlist__N, max_step=max_step, args=(H_matrix, dissipation_channels))

    full_dynamics = sol.y.T.reshape(time_steps, 2*N, 2*N)

    dynamics = [Qobj(i, dims=[[N, 2], [N, 2]]) for i in full_dynamics]

    return dynamics

def calculate_redfield_dynamics(parameters):

    #Retrieve needed parameters from the dictionary.
    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_detuning = parameters['qubit_detuning']
    cavity_drive_amplitude = parameters['cavity_drive_amplitude']
    cavity_detuning = parameters['cavity_detuning']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters["time_steps"]
    psi0_atom = parameters['initial_state_qubit']
    tlist = np.linspace(0, final_time__mus, time_steps)/(10**6)


    cavity_field = cavity_drive_amplitude/(-cavity_detuning + .5j*cavity_diss_rate)
    cavity_photon_number = np.real(cavity_field*np.conj(cavity_field))
    
    qubit_detuning_lamb_shift = qubit_detuning + eff_coupling*(2*cavity_photon_number + 1)
    diag_qubit_freq = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5

    coupling_sz = eff_coupling*qubit_detuning_lamb_shift/diag_qubit_freq
    coupling_sx = eff_coupling*rabi_freq/diag_qubit_freq

    #Norm everything
    tlist__N = tlist*norm
    cavity_detuning__N = cavity_detuning/norm
    cavity_drive_amplitude__N = cavity_drive_amplitude/norm
    qubit_detuning__N = qubit_detuning/norm
    eff_coupling__N = eff_coupling/norm
    rabi_freq__N = rabi_freq/norm
    cavity_diss_rate__N = cavity_diss_rate/norm
    qubit_detuning_lamb_shift__N = qubit_detuning_lamb_shift/norm
    diag_qubit_freq__N = diag_qubit_freq/norm
    coupling_sz__N = coupling_sz/norm
    coupling_sx__N = coupling_sx/norm



    ### Qubit Hamiltonian ###
    sz = sigmaz()
    sx = sigmax()
    sm = sigmam()

    sz_matrix = sz.full()
    sx_matrix = sx.full()
    sm_matrix = sm.full()
    H_qubit_matrix = 5*diag_qubit_freq__N*sz_matrix

    ### Redfield operators ###
    u1 = (cavity_photon_number*coupling_sz__N/(-1j*cavity_detuning__N + cavity_diss_rate__N/2)*sz_matrix \
        +  cavity_photon_number*coupling_sx__N/(1j*(-diag_qubit_freq__N - cavity_detuning__N) + cavity_diss_rate__N/2)*sm_matrix.T \
        +  cavity_photon_number*coupling_sx__N/(1j*(diag_qubit_freq__N - cavity_detuning__N) + cavity_diss_rate__N/2)*sm_matrix )

    s1 = coupling_sz__N*sz_matrix + coupling_sx__N*sx_matrix

    theta = np.arctan2(rabi_freq, qubit_detuning_lamb_shift)

    U = (1j * theta/2 * sigmay()).expm()

    rho0 = U*psi0_atom@psi0_atom.dag()*U.dag()
    rho0_matrix = rho0.full()
    rho0_vectorized = rho0_matrix.flatten()
    max_step =  0.05/np.max([diag_qubit_freq__N, coupling_sz__N, coupling_sx__N,])

    sol = solve_ivp(redfield_master_equation, (tlist__N[0], tlist__N[-1]), rho0_vectorized, method="DOP853",
                    t_eval=tlist__N, max_step=max_step, args=(H_qubit_matrix, [u1], [s1]))

    full_dynamics = sol.y.T.reshape(time_steps, 2, 2)

    dynamics = [U.dag()*Qobj(i, dims=[[2], [2]])*U for i in full_dynamics]

    return dynamics


def tdep_redfield_master_equation(t, rho_vec, hamiltonian, parameters, sz_matrix, sm_matrix, S_operators):

    Z_operators = tdep_Z_operators(parameters, sz_matrix, sm_matrix, t)

    return redfield_master_equation(t, rho_vec, hamiltonian, Z_operators, S_operators)

def tdep_Z_operators(parameters, sz_matrix, sm_matrix, t):
    Zop_term1__N = parameters["Zop_term1__N"]
    Zop_term2__N = parameters["Zop_term2__N"]
    Zop_term3__N = parameters["Zop_term3__N"]
    diag_qubit_freq__N = parameters["diag_qubit_freq__N"]
    cavity_diss_rate__N = parameters["cavity_diss_rate__N"]
    cavity_detuning__N = parameters["cavity_detuning__N"]

    term1 = Zop_term1__N*(np.exp(1j*cavity_detuning__N*t - cavity_diss_rate__N*t)-1)*sz_matrix
    term2 = Zop_term2__N*(np.exp(1j*(diag_qubit_freq__N+cavity_detuning__N)*t - cavity_diss_rate__N*t)-1)*sm_matrix.T
    term3 = Zop_term3__N*(np.exp(1j*(-diag_qubit_freq__N+cavity_detuning__N)*t - cavity_diss_rate__N*t)-1)*sm_matrix

    u1 =  term1 + term2 + term3

    return [u1]



def calculate_tdep_redfield_dynamics(parameters):

    #Retrieve needed parameters from the dictionary.
    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_detuning = parameters['qubit_detuning']
    cavity_drive_amplitude = parameters['cavity_drive_amplitude']
    cavity_detuning = parameters['cavity_detuning']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters["time_steps"]
    psi0_atom = parameters['initial_state_qubit']
    tlist = np.linspace(0, final_time__mus, time_steps)/(10**6)


    cavity_field = cavity_drive_amplitude/(-cavity_detuning + .5j*cavity_diss_rate)
    cavity_photon_number = np.real(cavity_field*np.conj(cavity_field))
    
    qubit_detuning_lamb_shift = qubit_detuning + eff_coupling*(2*cavity_photon_number + 1)
    diag_qubit_freq = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5

    diag_qubit_freq = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
    coupling_sz = eff_coupling*qubit_detuning_lamb_shift/diag_qubit_freq
    coupling_sx = eff_coupling*rabi_freq/diag_qubit_freq

    #Norm everything
    tlist__N = tlist*norm
    cavity_detuning__N = cavity_detuning/norm
    cavity_drive_amplitude__N = cavity_drive_amplitude/norm
    qubit_detuning__N = qubit_detuning/norm
    eff_coupling__N = eff_coupling/norm
    rabi_freq__N = rabi_freq/norm
    cavity_diss_rate__N = cavity_diss_rate/norm
    qubit_detuning_lamb_shift__N = qubit_detuning_lamb_shift/norm
    diag_qubit_freq__N = diag_qubit_freq/norm
    coupling_sz__N = coupling_sz/norm
    coupling_sx__N = coupling_sx/norm

    parameters_red = parameters
    parameters_red["diag_qubit_freq__N"] = diag_qubit_freq__N
    parameters_red["coupling_sz__N"] = coupling_sz__N
    parameters_red["coupling_sx__N"] = coupling_sx__N
    parameters_red["cavity_detuning__N"] = cavity_detuning__N
    parameters_red["cavity_diss_rate__N"] = cavity_diss_rate__N

    ### Qubit Hamiltonian ###
    sz = sigmaz()
    sx = sigmax()
    sm = sigmam()

    sz_matrix = sz.full()
    sx_matrix = sx.full()
    sm_matrix = sm.full()
    H_qubit_matrix = .5*diag_qubit_freq__N*sz_matrix

    ### Redfield operators ###
    Zop_term1__N = cavity_photon_number*coupling_sz__N/(1j*cavity_detuning__N - cavity_diss_rate__N/2)
    Zop_term2__N = cavity_photon_number*coupling_sx__N/(1j*(diag_qubit_freq__N + cavity_detuning__N) - cavity_diss_rate__N/2)
    Zop_term3__N = cavity_photon_number*coupling_sx__N/(1j*(diag_qubit_freq__N - cavity_detuning__N) - cavity_diss_rate__N/2)

    parameters_red["Zop_term1__N"]=Zop_term1__N
    parameters_red["Zop_term2__N"]=Zop_term2__N
    parameters_red["Zop_term3__N"]=Zop_term3__N

    s1 = coupling_sz__N*sz_matrix + coupling_sx__N*sx_matrix

    theta = np.arctan2(rabi_freq, qubit_detuning_lamb_shift)

    U = (1j * theta/2 * sigmay()).expm()

    rho0 = U*psi0_atom@psi0_atom.dag()*U.dag()
    rho0_matrix = rho0.full()
    rho0_vectorized = rho0_matrix.flatten()
    max_step =  0.05/np.max([diag_qubit_freq__N, coupling_sz__N, coupling_sx__N,])

    sol = solve_ivp(tdep_redfield_master_equation, (tlist__N[0], tlist__N[-1]), rho0_vectorized, method="DOP853",
                    t_eval=tlist__N, max_step=max_step, args=(H_qubit_matrix, parameters_red, sz_matrix, sm_matrix, [s1]))

    full_dynamics = sol.y.T.reshape(time_steps, 2, 2)

    dynamics = [U.dag()*Qobj(i, dims=[[2], [2]])*U for i in full_dynamics]
    return dynamics


def tdep_redfield_master_equation_polaron(t, rho_vec, hamiltonian, parameters, sz_matrix, sm_matrix, S_operators):

    Z_operators = tdep_Z_operators_polaron(parameters, sz_matrix, sm_matrix, t)

    return redfield_master_equation(t, rho_vec, hamiltonian, Z_operators, S_operators)

def calculate_tdep_redfield_dynamics_polaron(parameters):

    #Retrieve needed parameters from the dictionary.
    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_detuning = parameters['qubit_detuning']
    cavity_drive_amplitude = parameters['cavity_drive_amplitude']
    cavity_detuning = parameters['cavity_detuning']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters["time_steps"]
    psi0_atom = parameters['initial_state_qubit']
    target_state = parameters['target_state']
    tlist = np.linspace(0, final_time__mus, time_steps)/(10**6)



    cavity_field_e = cavity_drive_amplitude/(-(cavity_detuning + eff_coupling) + .5j*cavity_diss_rate)
    cavity_field_g = cavity_drive_amplitude/(-(cavity_detuning - eff_coupling) + .5j*cavity_diss_rate)


    #calculate initial state of the qubit according to herr Polaron

    H_qubit_target = .5*((qubit_detuning+eff_coupling*(2*np.real(cavity_field_g*np.conj(cavity_field_e))+ 1))*sigmaz() + rabi_freq*sigmax())
    evalues_target_pol, estates_target_pol = H_qubit_target.eigenstates()

    if target_state == 'up':
        target = estates_target_pol[0]
    elif target_state == 'down':
        target = estates_target_pol[1]

    z_exp_target = expect(sigmaz(), target)
    P_e = (1+z_exp_target)/2
    P_g = (1-z_exp_target)/2

    qubit_detuning_lamb_shift = qubit_detuning + eff_coupling*(2*(P_e*np.abs(cavity_field_e)**2 + P_g*np.abs(cavity_field_g))+ 1)

    diag_qubit_freq = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
    coupling_sz = eff_coupling*qubit_detuning_lamb_shift/diag_qubit_freq
    coupling_sx = eff_coupling*rabi_freq/diag_qubit_freq

    #Norm everything
    tlist__N = tlist*norm
    cavity_detuning__N = cavity_detuning/norm
    cavity_drive_amplitude__N = cavity_drive_amplitude/norm
    qubit_detuning__N = qubit_detuning/norm
    eff_coupling__N = eff_coupling/norm
    rabi_freq__N = rabi_freq/norm
    cavity_diss_rate__N = cavity_diss_rate/norm
    qubit_detuning_lamb_shift__N = qubit_detuning_lamb_shift/norm
    diag_qubit_freq__N = diag_qubit_freq/norm
    coupling_sz__N = coupling_sz/norm
    coupling_sx__N = coupling_sx/norm

    parameters_red = parameters
    parameters_red["diag_qubit_freq__N"] = diag_qubit_freq__N
    parameters_red["coupling_sz__N"] = coupling_sz__N
    parameters_red["coupling_sx__N"] = coupling_sx__N
    parameters_red["cavity_detuning__N"] = cavity_detuning__N
    parameters_red["cavity_diss_rate__N"] = cavity_diss_rate__N
    parameters_red["eff_coupling__N"] = eff_coupling__N

    ### Qubit Hamiltonian ###
    sz = sigmaz()
    sx = sigmax()
    sm = sigmam()

    sz_matrix = sz.full()
    sx_matrix = sx.full()
    sm_matrix = sm.full()
    H_qubit_matrix = .5*diag_qubit_freq__N*sz_matrix

    ### Redfield operators ###
    Zop_term1_e__N = P_e*np.abs(cavity_field_e)**2*coupling_sz__N/(1j*(cavity_detuning__N + eff_coupling__N) - cavity_diss_rate__N/2)
    Zop_term2_e__N = P_e*np.abs(cavity_field_e)**2*coupling_sx__N/(1j*(diag_qubit_freq__N + eff_coupling__N + cavity_detuning__N) - cavity_diss_rate__N/2)
    Zop_term3_e__N = P_e*np.abs(cavity_field_e)**2*coupling_sx__N/(1j*(diag_qubit_freq__N - eff_coupling__N - cavity_detuning__N) - cavity_diss_rate__N/2)

    Zop_term1_g__N = P_g*np.abs(cavity_field_g)**2*coupling_sz__N/(1j*(cavity_detuning__N - eff_coupling__N) - cavity_diss_rate__N/2)
    Zop_term2_g__N = P_g*np.abs(cavity_field_g)**2*coupling_sx__N/(1j*(diag_qubit_freq__N - eff_coupling__N + cavity_detuning__N) - cavity_diss_rate__N/2)
    Zop_term3_g__N = P_g*np.abs(cavity_field_g)**2*coupling_sx__N/(1j*(diag_qubit_freq__N + eff_coupling__N - cavity_detuning__N) - cavity_diss_rate__N/2)


    parameters_red["Zop_term1_e__N"]=Zop_term1_e__N
    parameters_red["Zop_term2_e__N"]=Zop_term2_e__N
    parameters_red["Zop_term3_e__N"]=Zop_term3_e__N

    parameters_red["Zop_term1_g__N"]=Zop_term1_g__N
    parameters_red["Zop_term2_g__N"]=Zop_term2_g__N
    parameters_red["Zop_term3_g__N"]=Zop_term3_g__N

    s1 = coupling_sz__N*sz_matrix + coupling_sx__N*sx_matrix

    theta = np.arctan2(rabi_freq, qubit_detuning_lamb_shift)

    U = (1j * theta/2 * sigmay()).expm()

    rho0 = U*psi0_atom@psi0_atom.dag()*U.dag()
    rho0_matrix = rho0.full()
    rho0_vectorized = rho0_matrix.flatten()
    max_step =  0.05/np.max([diag_qubit_freq__N, coupling_sz__N, coupling_sx__N,])

    sol = solve_ivp(tdep_redfield_master_equation_polaron, (tlist__N[0], tlist__N[-1]), rho0_vectorized, method="DOP853",
                    t_eval=tlist__N, max_step=max_step, args=(H_qubit_matrix, parameters_red, sz_matrix, sm_matrix, [s1]))

    full_dynamics = sol.y.T.reshape(time_steps, 2, 2)

    dynamics = [U.dag()*Qobj(i, dims=[[2], [2]])*U for i in full_dynamics]
    return dynamics

def tdep_Z_operators_polaron(parameters, sz_matrix, sm_matrix, t):
    Zop_term1_e__N = parameters["Zop_term1_e__N"]
    Zop_term2_e__N = parameters["Zop_term2_e__N"]
    Zop_term3_e__N = parameters["Zop_term3_e__N"]

    Zop_term1_g__N = parameters["Zop_term1_g__N"]
    Zop_term2_g__N = parameters["Zop_term2_g__N"]
    Zop_term3_g__N = parameters["Zop_term3_g__N"]


    diag_qubit_freq__N = parameters["diag_qubit_freq__N"]
    cavity_diss_rate__N = parameters["cavity_diss_rate__N"]
    cavity_detuning__N = parameters["cavity_detuning__N"]
    eff_coupling__N = parameters["eff_coupling__N"]

    term1_e = Zop_term1_e__N*(np.exp(1j*(cavity_detuning__N + eff_coupling__N)*t - cavity_diss_rate__N*t)-1)*sz_matrix
    term2_e = Zop_term2_e__N*(np.exp(1j*(diag_qubit_freq__N + eff_coupling__N + cavity_detuning__N)*t - cavity_diss_rate__N*t)-1)*sm_matrix.T
    term3_e = Zop_term3_e__N*(np.exp(1j*(-diag_qubit_freq__N+ eff_coupling__N + cavity_detuning__N)*t - cavity_diss_rate__N*t)-1)*sm_matrix

    term1_g = Zop_term1_g__N*(np.exp(1j*(cavity_detuning__N - eff_coupling__N)*t - cavity_diss_rate__N*t)-1)*sz_matrix
    term2_g = Zop_term2_g__N*(np.exp(1j*(diag_qubit_freq__N - eff_coupling__N + cavity_detuning__N)*t - cavity_diss_rate__N*t)-1)*sm_matrix.T
    term3_g = Zop_term3_g__N*(np.exp(1j*(-diag_qubit_freq__N - eff_coupling__N + cavity_detuning__N)*t - cavity_diss_rate__N*t)-1)*sm_matrix

    u1 =  term1_e + term2_e + term3_e + term1_g + term2_g + term3_g

    return [u1]

def calculate_cavity_drive_amplitude_normal_shift(parameters):
    """Calculate the cavity drive amplitude out of the normal cavity Ansatz, is just to get a realistic value for this parameter."""
    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters['time_steps']
    target_state = parameters['target_state']

    #Use normal cavity Ansatz to calculate the drive_cavity_amplitude. It does not affect is just to get a number for 
    #the Polaron shift and some initial state for the cavity.

    cavity_photon_number = 10**(input_power/10)
    diag_qubit_freq_normal_shift = (qubit_targetz**2 + rabi_freq**2)**.5
    if target_state == 'up':
        cavity_detuning_normal_shift = diag_qubit_freq_normal_shift
    elif target_state == 'down':
        cavity_detuning_normal_shift = -diag_qubit_freq_normal_shift

    cavity_drive_amplitude = np.real((cavity_photon_number * (cavity_detuning_normal_shift**2 + .25*cavity_diss_rate**2))**.5)

    parameters['cavity_drive_amplitude'] = cavity_drive_amplitude

    return parameters

def get_initial_cavity_state_normal_shift(parameters):
    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters['time_steps']
    target_state = parameters['target_state']
    cavity_drive_amplitude = parameters['cavity_drive_amplitude']

    diag_qubit_freq_normal_shift = (qubit_targetz**2 + rabi_freq**2)**.5
    if target_state == 'up':
        cavity_detuning_normal_shift = diag_qubit_freq_normal_shift
    elif target_state == 'down':
        cavity_detuning_normal_shift = -diag_qubit_freq_normal_shift

    cavity_field = cavity_drive_amplitude/(-cavity_detuning_normal_shift + .5j*cavity_diss_rate)
    psi0_cavity = coherent(N,cavity_field)
    parameters['initial_state_cavity'] = psi0_cavity

    return parameters

def get_initial_qubit_state(parameters):
    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters['time_steps']
    target_state = parameters['target_state']
    cavity_drive_amplitude = parameters['cavity_drive_amplitude']


    #calculate initial state of the qubit according to herr Polaron
    H_qubit_target = .5*((qubit_targetz)*sigmaz() + rabi_freq*sigmax())
    evalues_target_pol, estates_target_pol = H_qubit_target.eigenstates()

    if target_state == 'up':
        initial_state_qubit = estates_target_pol[1]
    elif target_state == 'down':
        initial_state_qubit = estates_target_pol[0]

    parameters['initial_state_qubit'] = initial_state_qubit

    return parameters

def calculate_parameters_polaron(parameters):

    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters['time_steps']
    target_state = parameters['target_state']
    psi0_cavity = parameters['initial_state_cavity']
    cavity_drive_amplitude = parameters['cavity_drive_amplitude']


    #Use polaron shift to calculate the qubit detuning to achieve our target frequency and
    #the cooling. Here we are making the assumption that the polaron shift is the best we can do.


    diag_qubit_freq = (qubit_targetz**2 + rabi_freq**2)**.5

    if target_state == 'up':
        cavity_detuning = diag_qubit_freq
    elif target_state == 'down':
        cavity_detuning = -diag_qubit_freq

    cavity_field_e = cavity_drive_amplitude/(-(cavity_detuning + eff_coupling) + .5j*cavity_diss_rate)
    cavity_field_g = cavity_drive_amplitude/(-(cavity_detuning - eff_coupling) + .5j*cavity_diss_rate)
    qubit_detuning = qubit_targetz - eff_coupling*(2*np.real(cavity_field_g*np.conj(cavity_field_e))+ 1)

    sz_anteil = qubit_targetz/diag_qubit_freq
    sx_anteil = rabi_freq/diag_qubit_freq

    parameters['qubit_detuning'] = qubit_detuning
    parameters['cavity_detuning'] = cavity_detuning
    parameters['diag_qubit_freq'] = diag_qubit_freq
    parameters['sz_anteil'] = sz_anteil
    parameters['sx_anteil'] = sx_anteil

    return parameters

def calculate_parameters_normal_shift(parameters):

    N = parameters['N']
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters['time_steps']
    target_state = parameters['target_state']
    cavity_drive_amplitude = parameters['cavity_drive_amplitude']

  
    diag_qubit_freq = (qubit_targetz**2 + rabi_freq**2)**.5
    if target_state == 'up':
        cavity_detuning = diag_qubit_freq
    elif target_state == 'down':
        cavity_detuning = -diag_qubit_freq

    cavity_field = cavity_drive_amplitude/(-cavity_detuning + .5j*cavity_diss_rate)
    cavity_photon_number = np.abs(cavity_field)**2
    qubit_detuning = qubit_targetz - eff_coupling*(2*cavity_photon_number + 1) 
    
    sz_anteil = qubit_targetz/diag_qubit_freq
    sx_anteil = rabi_freq/diag_qubit_freq
 
    parameters['qubit_detuning'] = qubit_detuning
    parameters['cavity_detuning'] = cavity_detuning
    parameters['diag_qubit_freq'] = diag_qubit_freq
    parameters['sz_anteil'] = sz_anteil
    parameters['sx_anteil'] = sx_anteil

    return parameters





