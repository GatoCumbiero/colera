import glob
from matplotlib import pyplot as plt
import os
import pickle
import numpy as np
from qutip import *

def load_all_dynamics(filepath):
    """
    Load multi-dynamics from saved file (now includes time-dependent Redfield)
    """
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"Multi-dynamics loaded from: {filepath}", flush=True)
    print(f"Description: {loaded_data['description']}", flush=True)
    print(f"Timestamp: {loaded_data['timestamp']}", flush=True)
    
    # Return all four dynamics types
    return (loaded_data['full_dynamics'], 
            loaded_data['redfield_dynamics'], 
            loaded_data['rates_dynamics'], 
            loaded_data['tdep_redfield_dynamics'],  # Added this line
            loaded_data['parameters'])

def plot_dynamics_with_params(tlist, dynamics_dict, parameters, max_time=None):
    """
    Modified version that includes parameter text in the plot
    with optional time range limitation
    """
    # Determine time range to plot
    if max_time is None:
        max_time = max(tlist)
    
    time_mask = tlist <= max_time
    tlist_plot = tlist[time_mask]
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    
    # Define colors for different methods
    colors = {
        'Hamiltonian 7': '#000080',
        'Redfield': '#008000', 
        'Rates': '#800000',
        'Time-dep Redfield': '#FFA500'
    }
    N = parameters["N"]
    
    # Plot each method
    for label, dynamics in dynamics_dict.items():
        color = colors.get(label, None)
        
        # Calculate expectations (handle both system and combined Hilbert spaces)
        if hasattr(dynamics[0], 'dims') and dynamics[0].dims == [[N,2],[N,2]]:
            # Combined Hilbert space (cavity + qubit)
            x_exp = expect(tensor(qeye(N), sigmax()), dynamics)
            z_exp = expect(tensor(qeye(N), sigmaz()), dynamics)
        else:
            # Qubit only
            x_exp = expect(sigmax(), dynamics)
            z_exp = expect(sigmaz(), dynamics)
        
        # Plot only up to max_time
        axes[0].plot(tlist_plot, x_exp[time_mask], color=color, label=label)
        axes[1].plot(tlist_plot, z_exp[time_mask], color=color, label=label)
    
    # Format σ_x plot
    axes[0].set_ylabel(r'$<\sigma_x>$')
    axes[0].legend()
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    # Format σ_z plot  
    axes[1].set_ylabel(r'$<\sigma_z>$')
    axes[1].set_xlabel('Time (μs)')
    axes[1].legend()
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    # Add parameter text (similar to your original code)
    rabi_freq = parameters.get('rabi_freq', 0)
    cavity_diss_rate = parameters.get('cavity_diss_rate', 0)
    eff_coupling = parameters.get('eff_coupling', 0)
    qubit_targetz = parameters.get('qubit_targetz', 0)

    params_text = []
    params_text.append(f"Rabi: {rabi_freq/(2*np.pi*1e6):.1f} MHz")
    params_text.append(f"κ: {cavity_diss_rate/(2*np.pi*1e6):.1f} MHz")
    params_text.append(r"$\chi$" + f": {eff_coupling/(2*np.pi*1e6):.1f} MHz")
    params_text.append(r"$\Delta_q+\chi (2\overline{n}+1)$" + f": {qubit_targetz/(2*np.pi*1e6):.1f} MHz")
    params_text.append(f"Time range: 0-{max_time} μs")

    plt.suptitle(' | '.join(params_text), y=0.98)
    plt.tight_layout()
    
    return fig, axes

def plot_all_saved_dynamics(saved_dir="./saved_dynamics", plots_dir="./plots"):
    """
    Process all saved dynamics files and generate plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Find all .pkl files in the saved dynamics directory
    pattern = os.path.join(saved_dir, "*.pkl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No .pkl files found in {saved_dir}")
        return
    
    print(f"Found {len(files)} dynamics files to plot")
    
    for i, filepath in enumerate(files):
        try:
            print(f"Processing {i+1}/{len(files)}: {os.path.basename(filepath)}")
            
            # Load dynamics from file
            full_dynamics, redfield_dynamics, rates_dynamics, tdep_redfield_dynamics, parameters = load_all_dynamics(filepath)
            
            # Create dynamics dictionary for plotting
            dynamics_dict = {
                'Hamiltonian 7': full_dynamics,
                'Redfield': redfield_dynamics,
                'Rates': rates_dynamics,
                'Time-dep Redfield': tdep_redfield_dynamics
            }
            
            # Create time array
            final_time__mus = parameters['final_time__mus']
            time_steps = parameters['time_steps']
            tlist = np.linspace(0, final_time__mus, time_steps)
            
            # Generate plot
            fig, axes = plot_dynamics_with_params(tlist, dynamics_dict, parameters, 2)
            
            # Extract parameters for filename
            filename = os.path.basename(filepath).replace('.pkl', '')
            eff_coupling = parameters.get('eff_coupling', 0) / (2*np.pi*1e6)  # Convert to MHz
            
            # Save plot
            plot_filename = f"plot_{filename}.png"
            plot_filepath = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close figure to free memory
            
            print(f"  Saved plot: {plot_filename}")
            
        except Exception as e:
            print(f"  Error processing {filepath}: {str(e)}")
            continue
    
    print(f"\nAll plots saved to: {plots_dir}")

# Run the script
if __name__ == "__main__":
    plot_all_saved_dynamics()