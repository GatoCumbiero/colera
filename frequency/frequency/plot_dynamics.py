""" Compare redfield with normal rates."""

import pickle
from datetime import datetime
import numpy as np
from qutip import *
import os
from matplotlib import pyplot as plt
import sys
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

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
    qubit_detuning = qubit_targetz - eff_coupling*(2*cavity_photon_number + 1) 
    qubit_detuning_lamb_shift = qubit_detuning + eff_coupling*(2*cavity_photon_number + 1)
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
    H_qubit = .5*(qubit_detuning__N + eff_coupling__N)*sz + .5*rabi_freq__N*sx
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

def target_hamiltonian(parameters):

    input_power = parameters['input_power']
    rabi_freq = parameters['rabi_freq']
    eff_coupling = parameters['eff_coupling']
    qubit_targetz = parameters['qubit_targetz']
    norm = parameters['norm']

    #Calculate important parameters
    cavity_photon_number = 10**(input_power/10)
    qubit_detuning = qubit_targetz - eff_coupling*(2*cavity_photon_number + 1) 
    qubit_detuning_lamb_shift = qubit_detuning + eff_coupling*(2*cavity_photon_number + 1)
    rabi_freq__N = rabi_freq/norm
    qubit_detuning_lamb_shift__N = qubit_detuning_lamb_shift/norm

    H_qubit_target = .5*(qubit_detuning_lamb_shift__N*sigmaz() + rabi_freq__N*sigmax())
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
    qubit_detuning = qubit_targetz - eff_coupling*(2*cavity_photon_number + 1) 
    qubit_detuning_lamb_shift = qubit_detuning + eff_coupling*(2*cavity_photon_number + 1)

    theta = np.arctan2(rabi_freq, qubit_detuning_lamb_shift)

    U = (1j * theta/2 * sigmay()).expm()

    return U


def save_full_dynamics(dynamics_data, save_dir="./saved_dynamics"):
    """
    Save all four types of dynamics along with parameters
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data to save
    data_to_save = {
        'full_dynamics': dynamics_data['full_dynamics'], 
        'parameters': dynamics_data['parameters'],
        'timestamp': timestamp,
        'description': "Full-dynamics sweep data"
    }
    
    # Add specific parameter value to filename based on what's being swept
    if 'eff_coupling_MHz' in dynamics_data:
        param_value = dynamics_data['eff_coupling_MHz']
        filename = f"eff_coupling_{param_value:.1f}MHz_{timestamp}.pkl"
        data_to_save['description'] = f"Full-dynamics for eff_coupling={param_value}MHz"
    elif 'cavity_diss_rate_MHz' in dynamics_data:
        param_value = dynamics_data['cavity_diss_rate_MHz']
        filename = f"kappa_{param_value:.1f}MHz_{timestamp}.pkl"
        data_to_save['description'] = f"Full-dynamics for kappa={param_value}MHz"
    else:
        filename = f"dynamics_{timestamp}.pkl"
    
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"  All dynamics saved to: {filepath}", flush=True)
    return filepath


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
            loaded_data['parameters'])



def plot_dynamics_comparison(tlist, dynamics_dict, params_text, N, max_time=None):
    """
    Plot comparison of different dynamics methods
    
    Parameters:
    tlist: time array
    dynamics_dict: dictionary with keys and corresponding dynamics objects
                  e.g., {'Hamiltonian 7': full_dynamics, 'Redfield': redfield_dynamics, ...}
    max_time: maximum time to plot (uses max(tlist) if None)
    """
    
    # Determine time range to plot
    if max_time is None:
        max_time = max(tlist)
    
    time_mask = tlist <= max_time
    tlist_plot = tlist[time_mask]
    
    # Create plot - now 3 subplots instead of 2
    fig, axes = plt.subplots(3, 1, figsize=(4, 8))
    
    # Define colors for different methods
    colors = {
        'Hamiltonian 7': '#000080',
        'Redfield': '#008000', 
        'Rates': '#800000',
        'Time-dep Redfield': '#FFA500'
    }
    
    # Plot each method
    for label, dynamics in dynamics_dict.items():
        color = colors.get(label, None)
        
        # Calculate expectations (handle both system and combined Hilbert spaces)
        if hasattr(dynamics[0], 'dims') and dynamics[0].dims == [[N,2],[N,2]]:
            # Combined Hilbert space (cavity + qubit)
            x_exp = expect(tensor(qeye(N), sigmax()), dynamics)
            z_exp = expect(tensor(qeye(N), sigmaz()), dynamics)
            # Calculate cavity photon number expectation
            cavity_number_op = tensor(destroy(N).dag() * destroy(N), qeye(2))
            cavity_exp = expect(cavity_number_op, dynamics)
        else:
            # Qubit only - no cavity information available
            x_exp = expect(sigmax(), dynamics)
            z_exp = expect(sigmaz(), dynamics)
            cavity_exp = np.zeros_like(x_exp)  # Placeholder
        
        # Plot qubit expectations
        axes[0].plot(tlist_plot, x_exp[time_mask], color=color, label=label)
        axes[1].plot(tlist_plot, z_exp[time_mask], color=color, label=label)
        
        # Plot cavity photon number (only for methods with cavity information)
        if hasattr(dynamics[0], 'dims') and dynamics[0].dims == [[N,2],[N,2]]:
            axes[2].plot(tlist_plot, cavity_exp[time_mask], color=color, label=label)
    
    # Format σ_x plot
    axes[0].set_ylabel(r'$\langle\sigma_x\rangle$')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    # Format σ_z plot  
    axes[1].set_ylabel(r'$\langle\sigma_z\rangle$')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    # Format cavity photon number plot
    axes[2].set_ylabel(r'$\langle a^\dagger a \rangle$')
    axes[2].set_xlabel('Time (μs)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(' | '.join(params_text), y=0.98)
    plt.tight_layout()
    
    return fig, axes





def plot_and_save(filepath, save_dir="./plots0_5", max_time=None):
    """
    Load a saved dynamics file and plot comparison between Redfield and Lindblad (normal) dynamics.
    """

    # Load data
    full_dynamics, parameters = load_all_dynamics(filepath)
    N = parameters['N']
    time_steps = parameters['time_steps']
    final_time__mus = parameters['final_time__mus']

    # Time axis (in microseconds)
    tlist = np.linspace(0, final_time__mus, time_steps)

    # For comparison: here you can add Redfield results if you have them loaded separately
    # e.g., redfield_dynamics = ...
    # For now, we'll just plot the saved full_dynamics (Lindblad) version
    dynamics_dict = {"Normal rates": full_dynamics}

    # Format parameters text for the title
    params_text = [
        f"Rabi: {parameters['rabi_freq']/(2*np.pi*1e6):.1f} MHz",
        f"κ: {parameters['cavity_diss_rate']/(2*np.pi*1e6):.1f} MHz",
        r"$\chi$: " + f"{parameters['eff_coupling']/(2*np.pi*1e6):.1f} MHz",
    ]

    # === Plot σ_x, σ_z, and photon number ===
    fig, axes = plt.subplots(3, 1, figsize=(4, 8))

    x_exp = expect(tensor(qeye(N), sigmax()), full_dynamics)
    z_exp = expect(tensor(qeye(N), sigmaz()), full_dynamics)
    cavity_number_op = tensor(destroy(N).dag() * destroy(N), qeye(2))
    cavity_exp = expect(cavity_number_op, full_dynamics)

    if max_time is None:
        max_time = final_time__mus
    mask = tlist <= max_time

    axes[0].plot(tlist[mask], x_exp[mask], color='navy', label='Normal rates')
    axes[0].set_ylabel(r'$\langle\sigma_x\rangle$')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(tlist[mask], z_exp[mask], color='darkred', label='Normal rates')
    axes[1].set_ylabel(r'$\langle\sigma_z\rangle$')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(tlist[mask], cavity_exp[mask], color='darkgreen', label='Photon number')
    axes[2].set_ylabel(r'$\langle a^\dagger a \rangle$')
    axes[2].set_xlabel("Time (μs)")
    axes[2].grid(alpha=0.3)

    plt.suptitle(" | ".join(params_text))
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # === Save figure ===
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_name = f"comparison_plot_eff_coupling_{parameters['eff_coupling']:.1f}MHz__{timestamp}.png"
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    print(f"✓ Saved plot to: {fig_path}")
    plt.close(fig)

    dominant_freq, period = analyze_sigma_x_oscillations(tlist[mask], x_exp[mask],
                                                     save_dir="./plots_fft_kappa0_5",
                                                     label="Full dynamics")

    return parameters['eff_coupling'], dominant_freq



def analyze_sigma_x_oscillations(tlist, x_exp, save_dir="./plots_fft_kappa0_5", label="Normal rates"):
    """
    Compute and plot Fourier spectrum of ⟨σ_x⟩(t),
    find main oscillation frequency, and mark its period in the time-domain plot.
    """

    os.makedirs(save_dir, exist_ok=True)

    # === Preprocess ===
    t = np.array(tlist)
    y = np.array(x_exp)
    y -= np.mean(y)  # remove DC component
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt  # sampling frequency (Hz)

    # === Fourier transform ===
    N = len(y)
    freqs = np.fft.rfftfreq(N, dt)
    fft_vals = np.fft.rfft(y)
    spectrum = np.abs(fft_vals)

    # === Find dominant frequency ===
    # Ignore DC (freq = 0)
    valid = freqs > 0
    peaks, _ = find_peaks(spectrum[valid], height=np.max(spectrum[valid])*0.1)
    dominant_freq = freqs[valid][peaks[np.argmax(spectrum[valid][peaks])]]
    period = 1 / dominant_freq  # in same time units as tlist (usually microseconds)

    print(f"Dominant oscillation frequency: {dominant_freq/1e6:.3f} MHz")
    print(f"Period: {period:.3e} μs")

    # === Plot Fourier spectrum ===
    fig_fft, ax_fft = plt.subplots(figsize=(5, 3))
    ax_fft.plot(freqs[valid]/1e6, spectrum[valid], color="purple")
    ax_fft.axvline(dominant_freq/1e6, color="red", ls="--", label=f"f = {dominant_freq/1e6:.3f} MHz")
    ax_fft.set_xlabel("Frequency (MHz)")
    ax_fft.set_ylabel("|FFT(σₓ)|")
    ax_fft.set_title(f"Fourier spectrum of ⟨σₓ⟩({label})")
    ax_fft.legend()
    ax_fft.grid(alpha=0.3)

    # === Plot σₓ dynamics with period markers ===
    fig_dyn, ax_dyn = plt.subplots(figsize=(6, 3))
    ax_dyn.plot(t, y + np.mean(x_exp), color="navy", label=r'$\langle\sigma_x\rangle$')
    for n in range(1, int(t[-1] / period) + 1):
        ax_dyn.axvline(n * period, color="red", ls="--", alpha=0.4)
    ax_dyn.set_xlabel("Time (μs)")
    ax_dyn.set_ylabel(r'$\langle\sigma_x\rangle$')
    ax_dyn.set_title(f"Oscillation period markers ({label})")
    ax_dyn.legend()
    ax_dyn.grid(alpha=0.3)

    # === Save both plots ===
    base_name = f"sigma_x_fft_{label.replace(' ', '_')}"
    fig_fft.savefig(os.path.join(save_dir, base_name + "_spectrum.png"), dpi=300, bbox_inches="tight")
    fig_dyn.savefig(os.path.join(save_dir, base_name + "_periods.png"), dpi=300, bbox_inches="tight")

    plt.close(fig_fft)
    plt.close(fig_dyn)

    return dominant_freq, period


if __name__ == "__main__":
    # Example: automatically plot all saved dynamics in a directory
    data_dir = "./kappa0_5"
    save_dir = "./plots_kappa0_5"

    eff_couplings = []
    dominant_freqs = []

    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            filepath = os.path.join(data_dir, file)
            try:
                eff_coupling, dominant_freq = plot_and_save(filepath, save_dir, max_time=1.5)
                eff_couplings.append(eff_coupling)
                dominant_freqs.append(dominant_freq)
            except Exception as e:
                print(f"✗ Failed to plot {file}: {e}")

    np.savez("./plots_kappa0_5/frequency_vs_coupling.npz",
         eff_couplings=eff_couplings,
         dominant_freqs=dominant_freqs)

    eff_couplings = np.array(eff_couplings)
    dominant_freqs = np.array(dominant_freqs)

    # === Convert units (Hz → MHz) ===
    eff_couplings_MHz = eff_couplings / (1e6)
    dominant_freqs_MHz = dominant_freqs  # if your FFT gives Hz; remove if already in MHz

    # === Plot frequency vs coupling ===
    plt.figure(figsize=(5, 3))
    plt.plot(eff_couplings_MHz, dominant_freqs_MHz, 'o-', color='purple')
    plt.xlabel(r'Effective coupling $\chi$ (MHz)')
    plt.ylabel('Dominant oscillation frequency (MHz)')
    plt.title('Dominant σₓ oscillation frequency vs effective coupling')
    plt.grid(alpha=0.3)

    freq_vs_coupling_path = os.path.join(save_dir, "freq_vs_eff_coupling.png")
    plt.savefig(freq_vs_coupling_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved summary plot to: {freq_vs_coupling_path}")


