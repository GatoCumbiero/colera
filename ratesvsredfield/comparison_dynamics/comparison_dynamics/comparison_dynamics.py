""" Compare redfield with normal rates."""

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
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters["time_steps"]
    psi0_atom = parameters['initial_state_qubit']

    #Calculate important parameters
    cavity_photon_number = 10**(input_power/10)
    qubit_detuning = qubit_targetz + eff_coupling*(2*cavity_photon_number + 1) 
    qubit_detuning_lamb_shift = qubit_detuning - eff_coupling*(2*cavity_photon_number + 1)
    cavity_detuning = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
    cavity_drive_amplitude = np.real((cavity_photon_number * (cavity_detuning**2 + .25*cavity_diss_rate**2))**.5)
    cavity_field = cavity_drive_amplitude/(-cavity_detuning + .5j*cavity_diss_rate)
    tlist = np.linspace(0, final_time__mus, time_steps)/(10**6)

    #Norm everything
    tlist__N = tlist*norm
    cavity_detuning__N = cavity_detuning/norm
    cavity_drive_amplitude__N = cavity_drive_amplitude/norm
    qubit_detuning__N = qubit_detuning/norm
    eff_coupling__N = eff_coupling/norm
    rabi_freq__N = rabi_freq/norm
    cavity_diss_rate__N = cavity_diss_rate/norm
    qubit_detuning_lamb_shift__N = qubit_detuning_lamb_shift/norm

    ### HAMILTONIAN 7 ####
    d = tensor(destroy(N), qeye(2))   
    sz = tensor(qeye(N), sigmaz())  
    sx = tensor(qeye(N), sigmax())     
    sm = tensor(qeye(N), sigmam()) 

    H_cav = cavity_detuning__N * d.dag()*d + cavity_drive_amplitude__N * (d + d.dag())
    H_qubit = -.5*(qubit_detuning__N - eff_coupling__N)*sz - .5*rabi_freq__N*sx
    H_int =   eff_coupling__N*d.dag()*d*sz
    H = H_cav + H_qubit + H_int

    H_matrix = H.full()
    d_matrix = d.full()


    dissipation_channels = [(cavity_diss_rate__N**.5*d_matrix)]

    psi0_cavity = coherent(N,cavity_field)
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
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters["time_steps"]
    psi0_atom = parameters['initial_state_qubit']

    #Calculate important parameters
    cavity_photon_number = 10**(input_power/10)
    qubit_detuning = qubit_targetz + eff_coupling*(2*cavity_photon_number + 1) 
    qubit_detuning_lamb_shift = qubit_detuning - eff_coupling*(2*cavity_photon_number + 1)
    cavity_detuning = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
    cavity_drive_amplitude = np.real((cavity_photon_number * (cavity_detuning**2 + .25*cavity_diss_rate**2))**.5)
    cavity_field = cavity_drive_amplitude/(-cavity_detuning + .5j*cavity_diss_rate)
    tlist = np.linspace(0, final_time__mus, time_steps)/(10**6)
    diag_qubit_freq = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
    coupling_sz = eff_coupling*qubit_detuning_lamb_shift/diag_qubit_freq
    coupling_sx = -eff_coupling*rabi_freq/diag_qubit_freq

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
    sm = sigmap()

    sz_matrix = sz.full()
    sx_matrix = sx.full()
    sm_matrix = sm.full()
    H_qubit_matrix = -.5*diag_qubit_freq__N*sz_matrix

    ### Redfield operators ###
    u1 = (cavity_photon_number*coupling_sz__N/(-1j*cavity_detuning__N + cavity_diss_rate__N/2)*sz_matrix \
        +  cavity_photon_number*coupling_sx__N/(1j*(-diag_qubit_freq__N - cavity_detuning__N) + cavity_diss_rate__N/2)*sm_matrix.T \
        +  cavity_photon_number*coupling_sx__N/(1j*(diag_qubit_freq__N - cavity_detuning__N) + cavity_diss_rate__N/2)*sm_matrix )

    s1 = coupling_sz__N*sz_matrix + coupling_sx__N*sx_matrix

    U = unitary_diag_qubit_hamiltonian(parameters)

    rho0 = U*psi0_atom@psi0_atom.dag()*U.dag()
    rho0_matrix = rho0.full()
    rho0_vectorized = rho0_matrix.flatten()
    max_step =  0.05/np.max([diag_qubit_freq__N, coupling_sz__N, coupling_sx__N,])

    sol = solve_ivp(redfield_master_equation, (tlist__N[0], tlist__N[-1]), rho0_vectorized, method="DOP853",
                    t_eval=tlist__N, max_step=max_step, args=(H_qubit_matrix, [u1], [s1]))

    full_dynamics = sol.y.T.reshape(time_steps, 2, 2)

    dynamics = [U.dag()*Qobj(i, dims=[[2], [2]])*U for i in full_dynamics]
    return dynamics


def target_hamiltonian(parameters):

    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']

    #Calculate important parameters
    cavity_photon_number = 10**(input_power/10)
    qubit_detuning = qubit_targetz + eff_coupling*(2*cavity_photon_number + 1) 
    qubit_detuning_lamb_shift = qubit_detuning - eff_coupling*(2*cavity_photon_number + 1)
    rabi_freq__N = rabi_freq/norm
    qubit_detuning_lamb_shift__N = qubit_detuning_lamb_shift/norm

    H_qubit_target = -.5*(qubit_detuning_lamb_shift__N*sigmaz() + rabi_freq__N*sigmax())
    return H_qubit_target

def unitary_diag_qubit_hamiltonian(parameters):
    """
    Returns U = exp(i * θ/2 * σ_y) that diagonlizes the qubit Hamiltonian
    """

    #Retrieve needed parameters from the dictionary.
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']


    #Calculate important parameters
    cavity_photon_number = 10**(input_power/10)
    qubit_detuning = qubit_targetz + eff_coupling*(2*cavity_photon_number + 1) 
    qubit_detuning_lamb_shift = qubit_detuning - eff_coupling*(2*cavity_photon_number + 1)

    theta = np.arctan2(rabi_freq, qubit_detuning_lamb_shift)

    U = (1j * theta/2 * sigmay()).expm()

    return U

def calculate_rates_dynamics(parameters):

    #Retrieve needed parameters from the dictionary.
    cavity_diss_rate = parameters['cavity_diss_rate']
    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']
    final_time__mus = parameters['final_time__mus'] 
    time_steps = parameters["time_steps"]
    psi0_atom = parameters['initial_state_qubit']


    #Calculate important parameters
    cavity_photon_number = 10**(input_power/10)
    qubit_detuning = qubit_targetz + eff_coupling*(2*cavity_photon_number + 1) 
    qubit_detuning_lamb_shift = qubit_detuning - eff_coupling*(2*cavity_photon_number + 1)
    cavity_detuning = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
    tlist = np.linspace(0, final_time__mus, time_steps)/(10**6)
    diag_qubit_freq = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
    theta = np.arctan2(rabi_freq,qubit_detuning_lamb_shift)
    S_nn_neg = (cavity_photon_number * cavity_diss_rate) / ((cavity_diss_rate/2)**2+(-rabi_freq/np.sin(theta) + cavity_detuning)**2)
    eff_qubit_excitation_rate = eff_coupling**2*S_nn_neg*np.sin(theta)**2
    S_nn_pos = (cavity_photon_number * cavity_diss_rate) / ((cavity_diss_rate/2)**2+(rabi_freq/np.sin(theta) + cavity_detuning)**2)
    eff_qubit_decay_rate = eff_coupling**2*S_nn_pos*np.sin(theta)**2 


    #Norm everything
    tlist__N = tlist*norm
    qubit_detuning_lamb_shift__N = qubit_detuning_lamb_shift/norm
    diag_qubit_freq__N = diag_qubit_freq/norm
    eff_qubit_excitation_rate__N = eff_qubit_excitation_rate/norm
    eff_qubit_decay_rate__N = eff_qubit_decay_rate/norm

    ### Qubit Hamiltonian ###
    sz = sigmaz()
    sx = sigmax()
    sm = sigmam()

    sz_matrix = sz.full()
    sx_matrix = sx.full()
    sm_matrix = sm.full()
    H_qubit_matrix = -.5*diag_qubit_freq__N*sz_matrix

    dissipation_channels = [eff_qubit_excitation_rate__N**.5*sm_matrix.T, eff_qubit_decay_rate__N**.5*sm_matrix]

    U = unitary_diag_qubit_hamiltonian(parameters)

    rho0 = U*psi0_atom@psi0_atom.dag()*U.dag()
    rho0_matrix = rho0.full()
    rho0_vectorized = rho0_matrix.flatten()
    max_step =  0.05/np.max([diag_qubit_freq__N, eff_qubit_excitation_rate__N, eff_qubit_decay_rate__N,])

    sol = solve_ivp(lindblad_master_equation, (tlist__N[0], tlist__N[-1]), rho0_vectorized, method="DOP853",
                     t_eval=tlist__N, max_step=max_step, args=(H_qubit_matrix, dissipation_channels))
    
    full_dynamics = sol.y.T.reshape(time_steps, 2, 2)

    dynamics = [U.dag()*Qobj(i, dims=[[2], [2]])*U for i in full_dynamics]
    return dynamics



def sweep_cavity_diss():
    """
    Main function to run dynamics for different cavity dissipation rates
    """
    # Base parameters
    N = 10
    input_power = 0
    rabi_freq = 2*np.pi*9*10**6
    eff_coupling = 2*np.pi*1*10**6
    qubit_targetz = 9*2*np.pi*10**6
    norm = eff_coupling
    final_time__mus = 20
    time_steps = 1000

    # Define cavity dissipation rates to sweep (in MHz, then convert to angular frequency)
    cavity_diss_rates_MHz = [1.0, 2.0, 3.0, 4.0, 4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    cavity_diss_rates = [2*np.pi * kappa * 10**6 for kappa in cavity_diss_rates_MHz]

    print(f"Running dynamics for {len(cavity_diss_rates)} different cavity dissipation rates:", flush=True)
    print(f"cavity_diss_rate values: {cavity_diss_rates_MHz} MHz", flush=True)
    print("=" * 70, flush=True)

    for i, cavity_diss_rate in enumerate(cavity_diss_rates):
        print(f"\nRun {i+1}/{len(cavity_diss_rates)}: cavity_diss_rate = {cavity_diss_rates_MHz[i]:.1f} MHz", flush=True)
        print("-" * 50, flush=True)

        # Set up parameters for this cavity dissipation rate
        parameters = {
            "N": N,
            "final_time__mus": final_time__mus,
            "time_steps": time_steps,
            "input_power": input_power,
            "rabi_freq": rabi_freq,
            "eff_coupling": eff_coupling,
            "qubit_targetz": qubit_targetz,
            "cavity_diss_rate": cavity_diss_rate,
            "norm": norm
        }

        # Calculate target Hamiltonian and initial state
        H_qubit_target = target_hamiltonian(parameters)
        eigenenergies, eigenstate = H_qubit_target.eigenstates()
        psi0_atom = eigenstate[-1]
        target_state = eigenstate[0]

        parameters["initial_state_qubit"] = psi0_atom

        try:
            # Calculate all three types of dynamics
            print("Calculating full dynamics...", flush=True)
            full_dynamics = calculate_full_dynamics(parameters)

            print("Calculating Redfield dynamics...", flush=True)
            redfield_dynamics = calculate_redfield_dynamics(parameters)

            print("Calculating rates dynamics...", flush=True)
            rates_dynamics = calculate_rates_dynamics(parameters)

            # Save all dynamics together using the same save function but different directory
            dynamics_data = {
                'full_dynamics': full_dynamics,
                'redfield_dynamics': redfield_dynamics,
                'rates_dynamics': rates_dynamics,
                'parameters': parameters.copy(),  # Make a copy to avoid reference issues
                'cavity_diss_rate_MHz': cavity_diss_rates_MHz[i]
            }

            # Use the same save function but with kappa-specific directory
            save_all_dynamics(dynamics_data, save_dir="./saved_kappa_sweep")
            
            print(f"✓ Successfully completed run {i+1}", flush=True)
            print(f"  cavity_diss_rate: {cavity_diss_rates_MHz[i]:.1f} MHz", flush=True)

        except Exception as e:
            print(f"✗ Error in run {i+1}: {str(e)}", flush=True)
            continue

    print("\n" + "=" * 70, flush=True)
    print("All cavity dissipation rate sweeps completed!", flush=True)


def save_all_dynamics(dynamics_data, save_dir="./saved_dynamics"):
    """
    Save all three types of dynamics along with parameters
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data to save
    data_to_save = {
        'full_dynamics': dynamics_data['full_dynamics'],
        'redfield_dynamics': dynamics_data['redfield_dynamics'],
        'rates_dynamics': dynamics_data['rates_dynamics'],
        'parameters': dynamics_data['parameters'],
        'timestamp': timestamp,
        'description': f"Multi-dynamics sweep data"
    }
    
    # Add specific parameter value to filename based on what's being swept
    if 'eff_coupling_MHz' in dynamics_data:
        param_value = dynamics_data['eff_coupling_MHz']
        filename = f"eff_coupling_{param_value:.1f}MHz_{timestamp}.pkl"
        data_to_save['description'] = f"Multi-dynamics for eff_coupling={param_value}MHz"
    elif 'cavity_diss_rate_MHz' in dynamics_data:
        param_value = dynamics_data['cavity_diss_rate_MHz']
        filename = f"kappa_{param_value:.1f}MHz_{timestamp}.pkl"
        data_to_save['description'] = f"Multi-dynamics for kappa={param_value}MHz"
    else:
        filename = f"dynamics_{timestamp}.pkl"
    
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"  All dynamics saved to: {filepath}", flush=True)
    return filepath


def load_all_dynamics(filepath):
    """
    Load multi-dynamics from saved file
    """
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"Multi-dynamics loaded from: {filepath}", flush=True)
    print(f"Description: {loaded_data['description']}", flush=True)
    print(f"Timestamp: {loaded_data['timestamp']}", flush=True)
    
    return (loaded_data['full_dynamics'], 
            loaded_data['redfield_dynamics'], 
            loaded_data['rates_dynamics'], 
            loaded_data['parameters'])

def sweep_eff_coupling():
    """
    Main function to run dynamics for different eff_coupling values
    """
    # Base parameters
    N = 10
    cavity_diss_rate = 2*np.pi*4.3*10**6
    input_power = 0
    rabi_freq = 2*np.pi*9*10**6
    qubit_targetz = 9*2*np.pi*10**6
    final_time__mus = 20
    time_steps = 1000

    # Define eff_coupling values to sweep (in MHz, then convert to angular frequency)
    eff_coupling_values_MHz = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    eff_coupling_values = [2*np.pi * ec * 10**6 for ec in eff_coupling_values_MHz]

    print(f"Running dynamics for {len(eff_coupling_values)} different eff_coupling values:", flush=True)
    print(f"eff_coupling values: {eff_coupling_values_MHz} MHz", flush=True)
    print("=" * 70, flush=True)

    for i, eff_coupling in enumerate(eff_coupling_values):
        norm = eff_coupling  # Use current eff_coupling as normalization

        print(f"\nRun {i+1}/{len(eff_coupling_values)}: eff_coupling = {eff_coupling_values_MHz[i]:.1f} MHz", flush=True)
        print("-" * 50, flush=True)

        # Set up parameters for this eff_coupling value
        parameters = {
            "N": N,
            "final_time__mus": final_time__mus,
            "time_steps": time_steps,
            "input_power": input_power,
            "rabi_freq": rabi_freq,
            "eff_coupling": eff_coupling,
            "qubit_targetz": qubit_targetz,
            "cavity_diss_rate": cavity_diss_rate,
            "norm": norm
        }

        # Calculate target Hamiltonian and initial state
        H_qubit_target = target_hamiltonian(parameters)
        eigenenergies, eigenstate = H_qubit_target.eigenstates()
        psi0_atom = eigenstate[-1]
        target_state = eigenstate[0]

        parameters["initial_state_qubit"] = psi0_atom

        try:
            # Calculate all three types of dynamics
            print("Calculating full dynamics...", flush=True)
            full_dynamics = calculate_full_dynamics(parameters)

            print("Calculating Redfield dynamics...", flush=True)
            redfield_dynamics = calculate_redfield_dynamics(parameters)

            print("Calculating rates dynamics...", flush=True)
            rates_dynamics = calculate_rates_dynamics(parameters)

            # Save all dynamics together
            dynamics_data = {
                'full_dynamics': full_dynamics,
                'redfield_dynamics': redfield_dynamics,
                'rates_dynamics': rates_dynamics,
                'parameters': parameters.copy(),  # Make a copy to avoid reference issues
                'eff_coupling_MHz': eff_coupling_values_MHz[i]
            }

            # Create a custom save function for this data structure
            save_all_dynamics(dynamics_data)
            
            print(f"✓ Successfully completed run {i+1}", flush=True)
            print(f"  eff_coupling: {eff_coupling_values_MHz[i]:.1f} MHz", flush=True)

        except Exception as e:
            print(f"✗ Error in run {i+1}: {str(e)}", flush=True)
            continue

    print("\n" + "=" * 70, flush=True)
    print("All eff_coupling sweeps completed!", flush=True)


if __name__ == "__main__":
    sweep_cavity_diss()