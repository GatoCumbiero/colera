import pickle
from datetime import datetime
import numpy as np
from qutip import *
import os
from matplotlib import pyplot as plt
import sys
from scipy.integrate import solve_ivp
import sandbox_correct as sb
import importlib
importlib.reload(sb)
import glob
import re
def analyze_coupling_dependence(folder_path):
    """
    Analyze all dynamics files in a folder and compute trace distances and fidelities
    vs coupling strength
    """
    # Find all pickle files in the folder
    pattern = os.path.join(folder_path, "eff_coupling_*.pkl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    # Extract coupling values and sort files
    coupling_data = []
    for filepath in files:
        # Extract coupling value from filename
        match = re.search(r'eff_coupling_([\d.]+)MHz\.pkl', filepath)
        if match:
            coupling = float(match.group(1))
            coupling_data.append((coupling, filepath))
    
    # Sort by coupling strength
    coupling_data.sort(key=lambda x: x[0])
    
    # Initialize results arrays
    couplings = []
    fid_normal = []
    fid_polaron = []
    trace_normal = []
    trace_polaron = []
    
    # Analyze each file using your code structure
    for coupling, filepath in coupling_data:
        
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)


            # Return all four dynamics types
            dynamics_normal_shift = loaded_data['full_dynamics_normal_shift']
            dynamics_polaron_shift = loaded_data['full_dynamics_polaron_shift']
            parameters = loaded_data['parameters']

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
            tlist = np.linspace(0, final_time__mus, time_steps)

            cavity_photon_number = 10**(input_power/10)
            qubit_detuning = qubit_targetz - eff_coupling*(2*cavity_photon_number + 1) 
            qubit_detuning_lamb_shift = qubit_detuning + eff_coupling*(2*cavity_photon_number + 1)
            cavity_detuning = (qubit_detuning_lamb_shift**2 + rabi_freq**2)**.5
            cavity_drive_amplitude = np.real((cavity_photon_number * (cavity_detuning**2 + .25*cavity_diss_rate**2))**.5)
            cavity_field = cavity_drive_amplitude/(-cavity_detuning + .5j*cavity_diss_rate)

            H_qubit = sb.target_hamiltonian(parameters)
            evalues_target, estates_target = H_qubit.eigenstates()

            # Get final qubit state from normal shift dynamics
            final_qubit_state_normal = ptrace(dynamics_normal_shift[-1], 1)
            
            # Calculate metrics for normal shift target
            fid_norm = fidelity(estates_target[0], final_qubit_state_normal)
            trace_norm = tracedist(estates_target[0], final_qubit_state_normal)

            H_qubit_pol = sb.target_hamiltonian_polaron(parameters)
            evalues_target_pol, estates_target_pol = H_qubit_pol.eigenstates()

            # Calculate metrics for polaron shift target
            fid_pol = fidelity(estates_target_pol[0], final_qubit_state_normal)
            trace_pol = tracedist(estates_target_pol[0], final_qubit_state_normal)

            # Store results
            couplings.append(coupling)
            fid_normal.append(fid_norm)
            fid_polaron.append(fid_pol)
            trace_normal.append(trace_norm)
            trace_polaron.append(trace_pol)
            

            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            continue
    
    return {
        'couplings': np.array(couplings),
        'fidelity_normal': np.array(fid_normal),
        'fidelity_polaron': np.array(fid_polaron),
        'trace_dist_normal': np.array(trace_normal),
        'trace_dist_polaron': np.array(trace_polaron),
        'parameters': parameters
    }

def plot_coupling_dependence(results):
    """Plot trace distance and fidelity vs coupling strength"""
    if results is None:
        print("No results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
    
    # Plot fidelity
    ax1.plot(results['couplings'], results['fidelity_normal'], 'o-', 
             label='Normal shift target', linewidth=2, markersize=6)
    ax1.plot(results['couplings'], results['fidelity_polaron'], 's-', 
             label='Polaron shift target', linewidth=2, markersize=6)
    ax1.set_ylabel('Fidelity', fontsize=12)
    # ax1.set_ylim(0, 1)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Fidelity vs Coupling Strength', fontsize=14)
    
    # Plot trace distance
    ax2.plot(results['couplings'], results['trace_dist_normal'], 'o-', 
             label='Normal shift target', linewidth=2, markersize=6)
    ax2.plot(results['couplings'], results['trace_dist_polaron'], 's-', 
             label='Polaron shift target', linewidth=2, markersize=6)
    ax2.set_xlabel(r'Effective Coupling (2$\pi$MHz)', fontsize=12)
    ax2.set_ylabel('Trace Distance', fontsize=12)
    # ax2.set_ylim(0, 1)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Trace Distance vs Coupling Strength', fontsize=14)
    
    return fig


def analyze_cavity_diss_dependence(folder_path):
    """
    Analyze all dynamics files in a folder and compute trace distances and fidelities
    vs cavity_diss_rate
    """
    # Find all pickle files in the folder
    pattern = os.path.join(folder_path, "cavity_diss_rate_*.pkl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    # Extract cavity_diss_rate values and sort files
    cavity_diss_data = []
    for filepath in files:
        # Extract cavity_diss_rate value from filename
        match = re.search(r'cavity_diss_rate_([\d.]+)MHz\.pkl', filepath)
        if match:
            cavity_diss_rate = float(match.group(1))
            cavity_diss_data.append((cavity_diss_rate, filepath))
    
    # Sort by cavity_diss_rate
    cavity_diss_data.sort(key=lambda x: x[0])
    
    # Initialize results arrays
    cavity_diss_rates = []
    fid_normal = []
    fid_polaron = []
    trace_normal = []
    trace_polaron = []
    
    # Analyze each file
    for cavity_diss_rate, filepath in cavity_diss_data:        
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)


            # Extract dynamics and parameters
            dynamics_normal_shift = loaded_data['full_dynamics_normal_shift']
            dynamics_polaron_shift = loaded_data['full_dynamics_polaron_shift']
            parameters = loaded_data['parameters']

            # Recreate target Hamiltonians
            H_qubit = sb.target_hamiltonian(parameters)
            evalues_target, estates_target = H_qubit.eigenstates()

            # Get final qubit state from normal shift dynamics
            final_qubit_state_normal = ptrace(dynamics_normal_shift[-1], 1)
            
            # Calculate metrics for normal shift target
            fid_norm = fidelity(estates_target[0], final_qubit_state_normal)
            trace_norm = tracedist(estates_target[0], final_qubit_state_normal)

            H_qubit_pol = sb.target_hamiltonian_polaron(parameters)
            evalues_target_pol, estates_target_pol = H_qubit_pol.eigenstates()

            # Calculate metrics for polaron shift target
            fid_pol = fidelity(estates_target_pol[0], final_qubit_state_normal)
            trace_pol = tracedist(estates_target_pol[0], final_qubit_state_normal)

            # Store results
            cavity_diss_rates.append(cavity_diss_rate)
            fid_normal.append(fid_norm)
            fid_polaron.append(fid_pol)
            trace_normal.append(trace_norm)
            trace_polaron.append(trace_pol)

            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            continue
    
    return {
        'cavity_diss_rates': np.array(cavity_diss_rates),
        'fidelity_normal': np.array(fid_normal),
        'fidelity_polaron': np.array(fid_polaron),
        'trace_dist_normal': np.array(trace_normal),
        'trace_dist_polaron': np.array(trace_polaron),
        'parameters': parameters
    }

def plot_cavity_diss_dependence(results):
    """Plot trace distance and fidelity vs cavity_diss_rate"""
    if results is None:
        print("No results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot fidelity
    ax1.plot(results['cavity_diss_rates'], results['fidelity_normal'], 'o-', 
             label='Normal shift target', linewidth=2, markersize=6)
    ax1.plot(results['cavity_diss_rates'], results['fidelity_polaron'], 's-', 
             label='Polaron shift target', linewidth=2, markersize=6)
    ax1.set_ylabel('Fidelity', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Fidelity vs Cavity Dissipation Rate', fontsize=14)
    
    # Plot trace distance
    ax2.plot(results['cavity_diss_rates'], results['trace_dist_normal'], 'o-', 
             label='Normal shift target', linewidth=2, markersize=6)
    ax2.plot(results['cavity_diss_rates'], results['trace_dist_polaron'], 's-', 
             label='Polaron shift target', linewidth=2, markersize=6)
    ax2.set_xlabel(r'Cavity Dissipation Rate (2$\pi$MHz)', fontsize=12)
    ax2.set_ylabel('Trace Distance', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Trace Distance vs Cavity Dissipation Rate', fontsize=14)
    
    return fig


def analyze_qubit_targetz_dependence(folder_path):
    """
    Analyze all dynamics files in a folder and compute trace distances and fidelities
    vs qubit_targetz for both normal and polaron targets
    """
    # Find all pickle files in the folder
    pattern = os.path.join(folder_path, "qubit_targetz_*.pkl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    # Extract qubit_targetz values and sort files
    qubit_targetz_data = []
    for filepath in files:
        # Extract qubit_targetz value from filename
        match = re.search(r'qubit_targetz_([-\d.]+)MHz\.pkl', filepath)
        if match:
            qubit_targetz = float(match.group(1))
            qubit_targetz_data.append((qubit_targetz, filepath))
    
    # Sort by qubit_targetz
    qubit_targetz_data.sort(key=lambda x: x[0])
    
    # Initialize results arrays
    qubit_targetz_rates = []
    fid_normal = []
    fid_polaron = []
    trace_normal = []
    trace_polaron = []
    
    # Analyze each file
    for qubit_targetz, filepath in qubit_targetz_data:
        
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)



            # Extract dynamics and parameters
            dynamics_normal_shift = loaded_data['full_dynamics_normal_shift']
            parameters = loaded_data['parameters']

            # Recreate target Hamiltonians
            H_qubit = sb.target_hamiltonian(parameters)
            evalues_target, estates_target = H_qubit.eigenstates()

            H_qubit_pol = sb.target_hamiltonian_polaron(parameters)
            evalues_target_pol, estates_target_pol = H_qubit_pol.eigenstates()

            # Get final qubit state from normal shift dynamics
            final_qubit_state_normal = ptrace(dynamics_normal_shift[-1], 1)
            
            # Calculate metrics for normal shift target
            fid_norm = fidelity(estates_target[0], final_qubit_state_normal)
            trace_norm = tracedist(estates_target[0], final_qubit_state_normal)

            # Calculate metrics for polaron shift target
            fid_pol = fidelity(estates_target_pol[0], final_qubit_state_normal)
            trace_pol = tracedist(estates_target_pol[0], final_qubit_state_normal)

            # Store results
            qubit_targetz_rates.append(qubit_targetz)
            fid_normal.append(fid_norm)
            fid_polaron.append(fid_pol)
            trace_normal.append(trace_norm)
            trace_polaron.append(trace_pol)
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            continue
    
    return {
        'qubit_targetz_rates': np.array(qubit_targetz_rates),
        'fidelity_normal': np.array(fid_normal),
        'fidelity_polaron': np.array(fid_polaron),
        'trace_dist_normal': np.array(trace_normal),
        'trace_dist_polaron': np.array(trace_polaron),
        'parameters': parameters
    }

def plot_qubit_targetz_dependence(results):
    """Plot trace distance and fidelity vs qubit_targetz for both normal and polaron"""
    if results is None:
        print("No results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot fidelity
    ax1.plot(results['qubit_targetz_rates'], results['fidelity_normal'], 'o-', 
             label='Normal shift target', linewidth=2, markersize=6)
    ax1.plot(results['qubit_targetz_rates'], results['fidelity_polaron'], 's-', 
             label='Polaron shift target', linewidth=2, markersize=6)
    ax1.set_ylabel('Fidelity', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Fidelity vs Qubit Target Z Frequency', fontsize=14)
    
    # Plot trace distance
    ax2.plot(results['qubit_targetz_rates'], results['trace_dist_normal'], 'o-', 
             label='Normal shift target', linewidth=2, markersize=6)
    ax2.plot(results['qubit_targetz_rates'], results['trace_dist_polaron'], 's-', 
             label='Polaron shift target', linewidth=2, markersize=6)
    ax2.set_xlabel('Qubit Target Z Frequency (MHz)', fontsize=12)
    ax2.set_ylabel('Trace Distance', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Trace Distance vs Qubit Target Z Frequency', fontsize=14)

    return fig