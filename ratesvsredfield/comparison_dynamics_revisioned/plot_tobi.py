import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from qutip import *
import glob

def load_and_analyze_dynamics(save_dir="./saved_dynamics"):
    """Load all saved dynamics files and analyze them"""
    
    # Find all saved dynamics files
    pattern = os.path.join(save_dir, "eff_coupling_*.pkl")
    files = glob.glob(pattern)
    files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(files)} dynamics files")
    
    # Initialize data structures
    eff_coupling_values = []
    time_data = []
    
    # For storing distances for each method
    redfield_distances = []
    lindblad_distances = []
    tdep_redfield_distances = []
    
    # For storing final distances vs gamma/Omega
    final_redfield_dist = []
    final_lindblad_dist = []
    final_tdep_redfield_dist = []
    gamma_over_omega_values = []
    
    # Store parameters for titles
    kappa_values = []
    targetz_values = []
    
    for file in files:
        try:
            # Load the dynamics data
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract dynamics and parameters
            full_dynamics = data['full_dynamics']  # "exact"
            redfield_dynamics = data['redfield_dynamics']
            rates_dynamics = data['rates_dynamics']  # Lindblad
            tdep_redfield_dynamics = data['tdep_redfield_dynamics']  # Tdep_redfield
            parameters = data['parameters']
            eff_coupling_MHz = data.get('eff_coupling_MHz', parameters['eff_coupling'] / (2*np.pi*1e6))
            
            # Calculate time array
            t_final = parameters['final_time__mus']
            time_steps = parameters['time_steps']
            tlist = np.linspace(0, t_final, time_steps)
            
            # Calculate gamma/Omega ratio (γ = eff_coupling, Ω = rabi_freq)
            gamma = parameters['eff_coupling'] / (2*np.pi*1e6)  # in MHz
            Omega = parameters['rabi_freq'] / (2*np.pi*1e6)  # in MHz
            gamma_over_omega = gamma / Omega
            
            # Store parameters for titles
            kappa = parameters['cavity_diss_rate'] / (2*np.pi*1e6)  # in MHz
            targetz = parameters['qubit_targetz'] / (2*np.pi*1e6)  # in MHz
            
            print(f"Processing: χ = {eff_coupling_MHz:.2f} MHz, χ/Ω = {gamma_over_omega:.3f}")
            
            # Calculate trace distances over time using QuTiP's tracedist
            redfield_dist = []
            lindblad_dist = []
            tdep_dist = []
            
            for i in range(len(full_dynamics)):
                # Use QuTiP's built-in trace distance function
                redfield_dist.append(tracedist(ptrace(full_dynamics[i],1), redfield_dynamics[i]))
                lindblad_dist.append(tracedist(ptrace(full_dynamics[i],1), rates_dynamics[i]))
                tdep_dist.append(tracedist(ptrace(full_dynamics[i],1), tdep_redfield_dynamics[i]))
            
            # Store the data
            eff_coupling_values.append(eff_coupling_MHz)
            time_data.append(tlist)
            redfield_distances.append(redfield_dist)
            lindblad_distances.append(lindblad_dist)
            tdep_redfield_distances.append(tdep_dist)
            
            # Store final distances for parameter sweep plot
            final_redfield_dist.append(redfield_dist[-1])
            final_lindblad_dist.append(lindblad_dist[-1])
            final_tdep_redfield_dist.append(tdep_dist[-1])
            gamma_over_omega_values.append(gamma_over_omega)
            kappa_values.append(kappa)
            targetz_values.append(targetz)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return {
        'eff_coupling_values': eff_coupling_values,
        'gamma_over_omega_values': gamma_over_omega_values,
        'time_data': time_data,
        'redfield_distances': redfield_distances,
        'lindblad_distances': lindblad_distances,
        'tdep_redfield_distances': tdep_redfield_distances,
        'final_redfield_dist': final_redfield_dist,
        'final_lindblad_dist': final_lindblad_dist,
        'final_tdep_redfield_dist': final_tdep_redfield_dist,
        'kappa_values': kappa_values,
        'targetz_values': targetz_values
    }

def plot_dynamics_comparison(analysis_data, selected_indices=None):
    """Plot dynamics comparison similar to the screenshot"""
    
    if selected_indices is None:
        # Select a few representative cases
        n_cases = min(3, len(analysis_data['eff_coupling_values']))
        selected_indices = np.linspace(0, len(analysis_data['eff_coupling_values'])-1, n_cases, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot (a): Time dynamics for selected gamma/Omega values
    ax1 = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, idx in enumerate(selected_indices):
        if idx < len(analysis_data['time_data']):
            t = analysis_data['time_data'][idx]
            # χt = eff_coupling * t (convert to microseconds and MHz)
            chi = analysis_data['eff_coupling_values'][idx]  # in MHz
            chi_t = chi * t  # χt (dimensionless)
            
            ax1.plot(chi_t, analysis_data['redfield_distances'][idx], 
                    color=colors[0], linestyle='-', alpha=0.8, label='Redfield' if i == 0 else "")
            ax1.plot(chi_t, analysis_data['lindblad_distances'][idx], 
                    color=colors[1], linestyle='--', alpha=0.8, label='Lindblad' if i == 0 else "")
            ax1.plot(chi_t, analysis_data['tdep_redfield_distances'][idx], 
                    color=colors[2], linestyle=':', alpha=0.8, label='Tdep_redfield' if i == 0 else "")
    
    ax1.set_xlabel(r'$\chi t$')
    ax1.set_ylabel(r'dist($\rho$, $\rho_{\mathrm{ex}}$)')
    ax1.legend()
    ax1.set_title('(a) Time Dynamics')
    ax1.grid(True, alpha=0.3)
    
    # Plot (b): Final distance vs gamma/Omega
    ax2 = axes[0, 1]
    gamma_omega = analysis_data['gamma_over_omega_values']
    
    # Sort by gamma/Omega for clean plotting
    sort_idx = np.argsort(gamma_omega)
    gamma_omega_sorted = np.array(gamma_omega)[sort_idx]
    
    ax2.semilogx(gamma_omega_sorted, np.array(analysis_data['final_redfield_dist'])[sort_idx], 
                'o-', color=colors[0], label='Redfield', markersize=6)
    ax2.semilogx(gamma_omega_sorted, np.array(analysis_data['final_lindblad_dist'])[sort_idx], 
                's--', color=colors[1], label='Lindblad', markersize=6)
    ax2.semilogx(gamma_omega_sorted, np.array(analysis_data['final_tdep_redfield_dist'])[sort_idx], 
                '^:', color=colors[2], label='Tdep_redfield', markersize=6)
    
    ax2.set_xlabel(r'$\chi / \Omega$')
    ax2.set_ylabel(r'dist($\rho$, $\rho_{\mathrm{ex}}$)')
    ax2.legend()
    ax2.set_title('(b) Parameter Dependence')
    ax2.grid(True, alpha=0.3)
    
    # Plot (c): Zoomed parameter dependence (0 to 0.4 as in screenshot)
    ax3 = axes[1, 0]
    mask = (np.array(gamma_omega_sorted) >= 0) & (np.array(gamma_omega_sorted) <= 0.4)
    
    if np.any(mask):
        ax3.plot(gamma_omega_sorted[mask], np.array(analysis_data['final_redfield_dist'])[sort_idx][mask], 
                'o-', color=colors[0], label='Redfield', markersize=6)
        ax3.plot(gamma_omega_sorted[mask], np.array(analysis_data['final_lindblad_dist'])[sort_idx][mask], 
                's--', color=colors[1], label='Lindblad', markersize=6)
        ax3.plot(gamma_omega_sorted[mask], np.array(analysis_data['final_tdep_redfield_dist'])[sort_idx][mask], 
                '^:', color=colors[2], label='Tdep_redfield', markersize=6)
    
    ax3.set_xlabel(r'$\chi / \Omega$')
    ax3.set_ylabel(r'dist($\rho$, $\rho_{\mathrm{ex}}$)')
    ax3.legend()
    ax3.set_title('(c) Low Coupling Regime')
    ax3.grid(True, alpha=0.3)
    
    # Plot (d): Method performance comparison
    ax4 = axes[1, 1]
    
    # Calculate which method performs best (lowest distance) for each parameter point
    gamma_omega_array = np.array(gamma_omega_sorted)
    redfield_array = np.array(analysis_data['final_redfield_dist'])[sort_idx]
    lindblad_array = np.array(analysis_data['final_lindblad_dist'])[sort_idx]
    tdep_array = np.array(analysis_data['final_tdep_redfield_dist'])[sort_idx]
    
    # Find best method for each point
    all_methods = np.column_stack([redfield_array, lindblad_array, tdep_array])
    best_method_indices = np.argmin(all_methods, axis=1)
    
    # Plot regions where each method is best
    methods = ['Redfield', 'Lindblad', 'Tdep_redfield']
    colors_methods = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, method in enumerate(methods):
        mask = best_method_indices == i
        if np.any(mask):
            ax4.semilogx(gamma_omega_array[mask], np.full(np.sum(mask), i), 
                        's', color=colors_methods[i], label=method, markersize=8, alpha=0.7)
    
    ax4.set_xlabel(r'$\chi / \Omega$')
    ax4.set_ylabel('Best Method')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(methods)
    ax4.legend()
    ax4.set_title('(d) Best Method by Parameter Regime')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_individual_time_series(analysis_data, indices=None):
    """Plot individual time series for specific parameter values"""
    
    if indices is None:
        indices = range(min(3, len(analysis_data['eff_coupling_values'])))
    
    n_plots = len(indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        if idx < len(analysis_data['time_data']):
            ax = axes[i]
            t = analysis_data['time_data'][idx]
            chi = analysis_data['eff_coupling_values'][idx]  # in MHz
            chi_t = chi * t  # χt (dimensionless)
            
            ax.plot(chi_t, analysis_data['redfield_distances'][idx], 
                   'b-', label='Redfield', linewidth=2)
            ax.plot(chi_t, analysis_data['lindblad_distances'][idx], 
                   'r--', label='Lindblad', linewidth=2)
            ax.plot(chi_t, analysis_data['tdep_redfield_distances'][idx], 
                   'g:', label='Tdep_redfield', linewidth=2)
            
            ax.set_xlabel(r'$\chi t$')
            ax.set_ylabel(r'dist($\rho$, $\rho_{\mathrm{ex}}$)')
            # Add parameter information to title
            kappa = analysis_data['kappa_values'][idx]
            targetz = analysis_data['targetz_values'][idx]
            ax.set_title(f'χ/Ω = {analysis_data["gamma_over_omega_values"][idx]:.3f}\nκ = {kappa:.1f} MHz, Δ$_q$ = {targetz:.1f} MHz')
            ax.set_xlim(0, 2)
            ax.legend()
            ax.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.show()


def plot_low_coupling_regime(analysis_data):
    """Plot only the low coupling regime with y-values multiplied by χ/Ω"""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Sort by gamma/Omega for clean plotting
    gamma_omega = analysis_data['gamma_over_omega_values']
    sort_idx = np.argsort(gamma_omega)
    gamma_omega_sorted = np.array(gamma_omega)[sort_idx]
    
    # Filter for low coupling regime (0 to 0.4 as in screenshot)
    mask = (np.array(gamma_omega_sorted) >= 0) & (np.array(gamma_omega_sorted) <= 0.4)
    
    if np.any(mask):
        # Multiply y-values by χ/Ω
        redfield_scaled = np.array(analysis_data['final_redfield_dist'])[sort_idx][mask] * gamma_omega_sorted[mask]
        lindblad_scaled = np.array(analysis_data['final_lindblad_dist'])[sort_idx][mask] * gamma_omega_sorted[mask]
        tdep_scaled = np.array(analysis_data['final_tdep_redfield_dist'])[sort_idx][mask] * gamma_omega_sorted[mask]
        
        ax.plot(gamma_omega_sorted[mask], redfield_scaled, 
                'o-', color=colors[0], label='Redfield', markersize=6, linewidth=2)
        ax.plot(gamma_omega_sorted[mask], lindblad_scaled, 
                's--', color=colors[1], label='Lindblad', markersize=6, linewidth=2)
        ax.plot(gamma_omega_sorted[mask], tdep_scaled, 
                '^:', color=colors[2], label='Tdep_redfield', markersize=6, linewidth=2)
    
    ax.set_xlabel(r'$\chi / \Omega$', fontsize=14)
    ax.set_ylabel(r'$(\chi / \Omega) \times \mathrm{dist}(\rho, \rho_{\mathrm{ex}})$', fontsize=14)
    ax.legend(fontsize=12)
    # Add parameter information to title
    kappa = analysis_data['kappa_values'][0] if analysis_data['kappa_values'] else 0
    targetz = analysis_data['targetz_values'][0] if analysis_data['targetz_values'] else 0
    ax.set_title(f'Low Coupling Regime\nκ = {kappa:.1f} MHz, Δ$_q$ = {targetz:.1f} MHz', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Improve tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_summary_table(analysis_data):
    """Create a summary table of the results including scaled values"""
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS (including scaled distances)")
    print("="*80)
    
    # Sort by gamma/Omega for clean display
    sort_idx = np.argsort(analysis_data['gamma_over_omega_values'])
    gamma_omega_sorted = np.array(analysis_data['gamma_over_omega_values'])[sort_idx]
    redfield_sorted = np.array(analysis_data['final_redfield_dist'])[sort_idx]
    lindblad_sorted = np.array(analysis_data['final_lindblad_dist'])[sort_idx]
    tdep_sorted = np.array(analysis_data['final_tdep_redfield_dist'])[sort_idx]
    
    # Calculate scaled distances
    redfield_scaled = redfield_sorted * gamma_omega_sorted
    lindblad_scaled = lindblad_sorted * gamma_omega_sorted
    tdep_scaled = tdep_sorted * gamma_omega_sorted
    
    print(f"{'χ/Ω':<8} {'Redfield':<10} {'R_scaled':<10} {'Lindblad':<10} {'L_scaled':<10} {'Tdep_redfield':<12} {'T_scaled':<10} {'Best':<10}")
    print("-" * 80)
    
    for i in range(len(gamma_omega_sorted)):
        dists = [redfield_sorted[i], lindblad_sorted[i], tdep_sorted[i]]
        best_idx = np.argmin(dists)
        best_method = ['Redfield', 'Lindblad', 'Tdep_redfield'][best_idx]
        
        print(f"{gamma_omega_sorted[i]:<8.3f} {redfield_sorted[i]:<10.4f} {redfield_scaled[i]:<10.4f} "
              f"{lindblad_sorted[i]:<10.4f} {lindblad_scaled[i]:<10.4f} "
              f"{tdep_sorted[i]:<12.4f} {tdep_scaled[i]:<10.4f} {best_method:<10}")
        
def plot_time_dynamics_with_expectations(analysis_data, save_dir = "./saved_dynamics", selected_indices=None, steady_state_threshold=1e-3):
    """Plot time dynamics with sigma_x and sigma_z expectation values above, showing a bit after steady state"""
    
    if selected_indices is None:
        # Select a few representative cases
        n_cases = min(3, len(analysis_data['eff_coupling_values']))
        selected_indices = np.linspace(0, len(analysis_data['eff_coupling_values'])-1, n_cases, dtype=int)
    
    n_cases = len(selected_indices)
    fig, axes = plt.subplots(3, n_cases, figsize=(5*n_cases, 12))
    
    if n_cases == 1:
        axes = axes.reshape(3, 1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Load dynamics files for expectation values
    pattern = os.path.join(save_dir, "eff_coupling_*.pkl")
    files = glob.glob(pattern)
    files.sort()
    
    for col, idx in enumerate(selected_indices):
        if idx < len(analysis_data['time_data']):
            # Get the corresponding file
            if idx < len(files):
                with open(files[idx], 'rb') as f:
                    data = pickle.load(f)
                
                # Extract dynamics
                full_dynamics = [ptrace(i,1) for i in data['full_dynamics']]  # "exact"
                redfield_dynamics = data['redfield_dynamics']
                rates_dynamics = data['rates_dynamics']  # Lindblad
                tdep_redfield_dynamics = data['tdep_redfield_dynamics']  # Tdep_redfield
                parameters = data['parameters']
                
                # Calculate time array
                t_final = parameters['final_time__mus']
                time_steps = parameters['time_steps']
                tlist = np.linspace(0, t_final, time_steps)
                
                # Calculate chi for time scaling
                chi = parameters['eff_coupling'] / (2*np.pi*1e6)  # in MHz
                chi_t = chi * tlist  # χt (dimensionless)

                # Calculate expectation values for each method
                # Exact (full dynamics)
                exact_sx = [expect(sigmax(), rho) for rho in full_dynamics]
                exact_sz = [expect(sigmaz(), rho) for rho in full_dynamics]
                
                # Redfield
                redfield_sx = [expect(sigmax(), rho) for rho in redfield_dynamics]
                redfield_sz = [expect(sigmaz(), rho) for rho in redfield_dynamics]
                
                # Lindblad (rates dynamics)
                lindblad_sx = [expect(sigmax(), rho) for rho in rates_dynamics]
                lindblad_sz = [expect(sigmaz(), rho) for rho in rates_dynamics]
                
                # Tdep_redfield
                tdep_sx = [expect(sigmax(), rho) for rho in tdep_redfield_dynamics]
                tdep_sz = [expect(sigmaz(), rho) for rho in tdep_redfield_dynamics]
                
                # Find when steady state is reached for exact dynamics
                # Use both sigma_x and sigma_z to detect steady state
                steady_state_reached = False
                steady_state_idx = len(exact_sx) - 1  # Default to last point
                
                # Check for steady state by looking at when derivatives become small
                for i in range(10, len(exact_sx) - 5):
                    # Calculate moving average of derivatives
                    window = 5
                    if i + window < len(exact_sx):
                        sx_deriv = np.abs(np.diff(exact_sx[i:i+window])).mean()
                        sz_deriv = np.abs(np.diff(exact_sz[i:i+window])).mean()
                        
                        # If both derivatives are below threshold, consider steady state reached
                        if sx_deriv < steady_state_threshold and sz_deriv < steady_state_threshold:
                            steady_state_idx = i + window
                            steady_state_reached = True
                            break
                
                # If no clear steady state found, use a more lenient approach
                if not steady_state_reached:
                    # Find when the values stop changing significantly
                    for i in range(len(exact_sx) - 10, 0, -1):
                        sx_change = np.abs(exact_sx[i] - exact_sx[-1])
                        sz_change = np.abs(exact_sz[i] - exact_sz[-1])
                        if sx_change > 0.1 or sz_change > 0.1:  # 10% of maximum change
                            steady_state_idx = min(i + 5, len(exact_sx) - 1)
                            break
                
                # Add a buffer after steady state (show ~10% more data after steady state)
                buffer_points = max(10, int(0.1 * steady_state_idx))  # At least 10 points, or 10% of steady state time
                end_idx = min(steady_state_idx + buffer_points, len(exact_sx) - 1)
                
                # Limit data to steady state region + buffer
                chi_t_limited = chi_t[:end_idx]
                exact_sx_limited = exact_sx[:end_idx]
                exact_sz_limited = exact_sz[:end_idx]
                redfield_sx_limited = redfield_sx[:end_idx]
                redfield_sz_limited = redfield_sz[:end_idx]
                lindblad_sx_limited = lindblad_sx[:end_idx]
                lindblad_sz_limited = lindblad_sz[:end_idx]
                tdep_sx_limited = tdep_sx[:end_idx]
                tdep_sz_limited = tdep_sz[:end_idx]
                
                # Plot sigma_x expectation values (top row)
                ax1 = axes[0, col]
                ax1.plot(chi_t_limited, exact_sx_limited, 'k-', label='Exact', linewidth=2)
                ax1.plot(chi_t_limited, redfield_sx_limited, color=colors[0], linestyle='-', label='Redfield', linewidth=1.5, alpha=0.8)
                ax1.plot(chi_t_limited, lindblad_sx_limited, color=colors[1], linestyle='--', label='Lindblad', linewidth=1.5, alpha=0.8)
                ax1.plot(chi_t_limited, tdep_sx_limited, color=colors[2], linestyle=':', label='Tdep_redfield', linewidth=1.5, alpha=0.8)
                ax1.set_ylabel(r'$\langle \sigma_x \rangle$', fontsize=12)
                if col == 0:
                    ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)
                # Add parameter information to title
                kappa = parameters['cavity_diss_rate'] / (2*np.pi*1e6)
                targetz = parameters['qubit_targetz'] / (2*np.pi*1e6)
                ax1.set_title(f'χ/Ω = {analysis_data["gamma_over_omega_values"][idx]:.3f}\nκ = {kappa:.1f} MHz, Δ$_q$ = {targetz:.1f} MHz', fontsize=12)
                
                # Plot sigma_z expectation values (middle row)
                ax2 = axes[1, col]
                ax2.plot(chi_t_limited, exact_sz_limited, 'k-', label='Exact', linewidth=2)
                ax2.plot(chi_t_limited, redfield_sz_limited, color=colors[0], linestyle='-', label='Redfield', linewidth=1.5, alpha=0.8)
                ax2.plot(chi_t_limited, lindblad_sz_limited, color=colors[1], linestyle='--', label='Lindblad', linewidth=1.5, alpha=0.8)
                ax2.plot(chi_t_limited, tdep_sz_limited, color=colors[2], linestyle=':', label='Tdep_redfield', linewidth=1.5, alpha=0.8)
                ax2.set_ylabel(r'$\langle \sigma_z \rangle$', fontsize=12)
                ax2.set_xlabel(r'$\chi t$', fontsize=12)
                if col == 0:
                    ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
            
            # Plot trace distances (bottom row) - also limited to steady state + buffer
            ax3 = axes[2, col]
            t = analysis_data['time_data'][idx]
            chi = analysis_data['eff_coupling_values'][idx]  # in MHz
            chi_t_full = chi * t  # χt (dimensionless)
            
            # Limit trace distance data to same steady state region + buffer
            chi_t_limited = chi_t_full[:end_idx]
            redfield_dist_limited = analysis_data['redfield_distances'][idx][:end_idx]
            lindblad_dist_limited = analysis_data['lindblad_distances'][idx][:end_idx]
            tdep_dist_limited = analysis_data['tdep_redfield_distances'][idx][:end_idx]
            
            ax3.plot(chi_t_limited, redfield_dist_limited, 
                   color=colors[0], linestyle='-', label='Redfield', linewidth=2, alpha=0.8)
            ax3.plot(chi_t_limited, lindblad_dist_limited, 
                   color=colors[1], linestyle='--', label='Lindblad', linewidth=2, alpha=0.8)
            ax3.plot(chi_t_limited, tdep_dist_limited, 
                   color=colors[2], linestyle=':', label='Tdep_redfield', linewidth=2, alpha=0.8)
            
            ax3.set_xlabel(r'$\chi t$', fontsize=12)
            ax3.set_ylabel(r'dist($\rho$, $\rho_{\mathrm{ex}}$)', fontsize=12)
            if col == 0:
                ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
    
    # Add overall title with parameter information
    # if analysis_data['kappa_values'] and analysis_data['targetz_values']:
    #     kappa = analysis_data['kappa_values'][0]
    #     targetz = analysis_data['targetz_values'][0]
    #     fig.suptitle(f'Time Dynamics: Expectation Values and Trace Distances\nκ = {kappa:.1f} MHz, Δ$_q$ = {targetz:.1f} MHz', fontsize=16, y=0.98)
    # else:
    #     fig.suptitle('Time Dynamics: Expectation Values and Trace Distances', fontsize=16, y=0.98)
    
    # Improve tick label size for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()
    
    return fig

# Replace the main execution to use the new function
if __name__ == "__main__":
    # Analyze all saved dynamics
    print("Loading and analyzing dynamics data...")
    savedir ="./saved_dynamics2"
    analysis_data = load_and_analyze_dynamics(savedir)
    
    if analysis_data['eff_coupling_values']:
        print(f"Successfully analyzed {len(analysis_data['eff_coupling_values'])} parameter sets")
        
        # Create the plot with expectation values and trace distances over time
        print("Creating plot with expectation values and trace distances...")
        fig = plot_time_dynamics_with_expectations(analysis_data, savedir, [1, 8, 16])
        
        # Also create the low coupling regime parameter sweep plot
        print("Creating low coupling regime parameter sweep plot...")
        fig2 = plot_low_coupling_regime(analysis_data)
        
        # Print summary table including scaled values
        create_summary_table(analysis_data)
        
        # Print overall statistics
        print("\n=== Overall Statistics ===")
        print(f"χ/Ω range: {np.min(analysis_data['gamma_over_omega_values']):.3f} - {np.max(analysis_data['gamma_over_omega_values']):.3f}")
        print(f"Redfield final distance: {np.mean(analysis_data['final_redfield_dist']):.4f} ± {np.std(analysis_data['final_redfield_dist']):.4f}")
        print(f"Lindblad final distance: {np.mean(analysis_data['final_lindblad_dist']):.4f} ± {np.std(analysis_data['final_lindblad_dist']):.4f}")
        print(f"Tdep_redfield final distance: {np.mean(analysis_data['final_tdep_redfield_dist']):.4f} ± {np.std(analysis_data['final_tdep_redfield_dist']):.4f}")
        
    else:
        print("No valid dynamics data found!")