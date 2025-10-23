import numpy as np
from qutip import *
import os
from matplotlib import pyplot as plt
import sys
from scipy.integrate import solve_ivp
from comparison import *
import pickle

def heatmap_diss_rate(cavity_diss_rate, save_path="./diss_rate_results"):
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    input_power_array = np.arange(10, -11, -2)
    qubit_targetz_array = np.arange(-20, 21, 2) * 2 * np.pi * 10**6  # -20 to 20 MHz in steps of 2 MHz
    
    qubit_targetz_mesh, input_power_mesh = np.meshgrid(qubit_targetz_array, input_power_array)

    dinput = len(input_power_array)
    dtarget = len(qubit_targetz_array)

    # Initialize result arrays
    heatmapx_target_state = np.empty_like(input_power_mesh, dtype=float)
    heatmapz_target_state = np.empty_like(input_power_mesh, dtype=float)

    final_fidelity_full_rates = np.empty_like(input_power_mesh, dtype=float)
    final_fidelity_full_redfield = np.empty_like(input_power_mesh, dtype=float)
    final_fidelity_full_target = np.empty_like(input_power_mesh, dtype=float)
    final_fidelity_redfield_target = np.empty_like(input_power_mesh, dtype=float)
    final_fidelity_rates_target = np.empty_like(input_power_mesh, dtype=float)

    avg_fidelity_full_rates = np.empty_like(input_power_mesh, dtype=float)
    avg_fidelity_full_redfield = np.empty_like(input_power_mesh, dtype=float)

    N = 10
    rabi_freq = 2*np.pi*9*10**6  # Fixed Rabi frequency
    eff_coupling = 2*np.pi*(1)*10**6
    norm = eff_coupling
    final_time__mus = 2
    time_steps = 1000

    parameters = {
        "N": N,
        "final_time__mus": final_time__mus,
        "time_steps": time_steps,
        "rabi_freq": rabi_freq,
        "eff_coupling": eff_coupling,
        "cavity_diss_rate": cavity_diss_rate,
        "norm": norm
    }

    # Progress tracking
    total_simulations = dinput * dtarget
    current_simulation = 0
    
    for i in range(dinput):
        parameters["input_power"] = input_power_mesh[i, 0]
        
        for j in range(dtarget):
            parameters["qubit_targetz"] = qubit_targetz_mesh[i, j]
            
            current_simulation += 1
            print(f"Progress: {current_simulation}/{total_simulations} "
                  f"(input power: {input_power_mesh[i, 0]} dB, "
                  f"target frequency: {qubit_targetz_mesh[i, j]/(2*np.pi*1e6):.1f} MHz)", flush=True)

            H_qubit_target = target_hamiltonian(parameters)
            eigenenergies, eigenstate = H_qubit_target.eigenstates()
            psi0_atom = eigenstate[-1]
            target_state = eigenstate[0]

            parameters["initial_state_qubit"] = psi0_atom

            # Calculate dynamics
            try:
                full_dynamics_not_trace = calculate_full_dynamics(parameters)
                full_dynamics = [ptrace(i,1) for i in full_dynamics_not_trace]
                redfield_dynamics = calculate_redfield_dynamics(parameters)
                rates_dynamics = calculate_rates_dynamics(parameters)

                # Store target state expectations
                heatmapx_target_state[i, j] = expect(sigmax(), target_state)
                heatmapz_target_state[i, j] = expect(sigmaz(), target_state)

                # Calculate fidelities
                final_fidelity_full_rates[i, j] = fidelity(full_dynamics[-1], rates_dynamics[-1])
                final_fidelity_full_redfield[i, j] = fidelity(full_dynamics[-1], redfield_dynamics[-1])
                final_fidelity_full_target[i, j] = fidelity(full_dynamics[-1], target_state)
                final_fidelity_redfield_target[i, j] = fidelity(redfield_dynamics[-1], target_state)
                final_fidelity_rates_target[i, j] = fidelity(rates_dynamics[-1], target_state)

                # Calculate average fidelities over time
                time_fidelities_full_rates = [fidelity(full_dynamics[k], rates_dynamics[k]) for k in range(len(full_dynamics))]
                time_fidelities_full_redfield = [fidelity(full_dynamics[k], redfield_dynamics[k]) for k in range(len(full_dynamics))]
                
                avg_fidelity_full_rates[i, j] = np.mean(time_fidelities_full_rates)
                avg_fidelity_full_redfield[i, j] = np.mean(time_fidelities_full_redfield)
                
            except Exception as e:
                print(f"Error at input_power={input_power_mesh[i, 0]}, target_freq={qubit_targetz_mesh[i, j]/(2*np.pi*1e6):.1f} MHz: {e}")
                # Fill with NaN in case of error
                heatmapx_target_state[i, j] = np.nan
                heatmapz_target_state[i, j] = np.nan
                final_fidelity_full_rates[i, j] = np.nan
                final_fidelity_full_redfield[i, j] = np.nan
                final_fidelity_full_target[i, j] = np.nan
                final_fidelity_redfield_target[i, j] = np.nan
                final_fidelity_rates_target[i, j] = np.nan
                avg_fidelity_full_rates[i, j] = np.nan
                avg_fidelity_full_redfield[i, j] = np.nan

    # Prepare results dictionary
    results = {
        'input_power_array': input_power_array,
        'qubit_targetz_array': qubit_targetz_array,
        'input_power_mesh': input_power_mesh,
        'qubit_targetz_mesh': qubit_targetz_mesh,
        'cavity_diss_rate': cavity_diss_rate,
        'rabi_freq': rabi_freq,
        'eff_coupling': eff_coupling,
        'final_fidelity_full_rates': final_fidelity_full_rates,
        'final_fidelity_full_redfield': final_fidelity_full_redfield,
        'final_fidelity_full_target': final_fidelity_full_target,
        'final_fidelity_redfield_target': final_fidelity_redfield_target,
        'final_fidelity_rates_target': final_fidelity_rates_target,
        'avg_fidelity_full_rates': avg_fidelity_full_rates,
        'avg_fidelity_full_redfield': avg_fidelity_full_redfield,
        'heatmapx_target_state': heatmapx_target_state,
        'heatmapz_target_state': heatmapz_target_state
    }

    # Save results
    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
    filename = f"rabi_sweep_kappa_{cavity_diss_rate/(2*np.pi*1e6):.1f}MHz_{timestamp}.npz"
    filepath = os.path.join(save_path, filename)
    
    np.savez(filepath, **results)
    
    # Also save as pickle for easier loading of QuTiP objects if needed
    pickle_filename = f"rabi_sweep_kappa_{cavity_diss_rate/(2*np.pi*1e6):.1f}MHz_{timestamp}.pkl"
    pickle_filepath = os.path.join(save_path, pickle_filename)
    
    with open(pickle_filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to:", flush=True)
    print(f"  {filepath}")
    print(f"  {pickle_filepath}")
    
    return results



def main():
    # Test with different cavity dissipation rates
    cavity_diss_rates = [2*np.pi*0.3*10**6]  # Add more rates if desired
    
    for rate in cavity_diss_rates:
        print(f"Calculating heatmap for κ = {rate/(2*np.pi*1e6):.1f} MHz", flush=True)
        results = heatmap_diss_rate(rate)
        
        # You can save results if needed
        # np.savez(f'heatmap_kappa_{rate/(2*np.pi*1e6):.1f}MHz.npz', **results)

if __name__ == "__main__":
    main()