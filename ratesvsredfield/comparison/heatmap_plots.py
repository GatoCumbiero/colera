import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import glob

def read_results(filepath):
    """
    Read saved results from either .npz or .pkl file
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        results = {key: data[key] for key in data.files}
        data.close()
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError("File must be either .npz or .pkl format")
    
    print(f"Loaded results from: {filepath}")
    return results

def find_all_results_files(directory):
    """
    Find all .npz and .pkl files in a directory
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    npz_files = glob.glob(os.path.join(directory, "*.npz"))
    pkl_files = glob.glob(os.path.join(directory, "*.pkl"))
    all_files = npz_files + pkl_files
    
    if not all_files:
        raise FileNotFoundError(f"No results files found in {directory}")
    
    print(f"Found {len(all_files)} result files:")
    for f in all_files:
        print(f"  {os.path.basename(f)}")
    
    return all_files

def plot_heatmaps(results, save_plots=False, plot_dir="./heatmap_plots", show_plot=True):
    """
    Plot comprehensive heatmaps from results data with proper figure management
    
    Parameters:
    - results: dictionary containing the results data
    - save_plots: whether to save the plots to files
    - plot_dir: directory to save plots (if save_plots=True)
    - show_plot: whether to display the plot on screen
    """
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Extract arrays from results
    input_power_array = results['input_power_array']
    qubit_targetz_array = results['qubit_targetz_array'] / (2 * np.pi * 1e6)
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(qubit_targetz_array, input_power_array)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Plot 1: Final Fidelity - Full vs Rates
    im1 = axes[0].pcolormesh(X, Y, results['final_fidelity_full_rates'], 
                            shading='auto', cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Final Fidelity: Full vs Rates')
    axes[0].set_ylabel('Input Power (dB)')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Final Fidelity - Full vs Redfield
    im2 = axes[1].pcolormesh(X, Y, results['final_fidelity_full_redfield'], 
                            shading='auto', cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Final Fidelity: Full vs Redfield')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Final Fidelity - Full vs Target
    im3 = axes[2].pcolormesh(X, Y, results['final_fidelity_full_target'], 
                            shading='auto', cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Final Fidelity: Full vs Target')
    plt.colorbar(im3, ax=axes[2])
    
    # Plot 4: Final Fidelity - Rates vs Target
    im4 = axes[3].pcolormesh(X, Y, results['final_fidelity_rates_target'], 
                            shading='auto', cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title('Final Fidelity: Rates vs Target')
    axes[3].set_xlabel('Target Frequency (MHz)')
    plt.colorbar(im4, ax=axes[3])
    
    # Plot 5: Average Fidelity - Full vs Rates
    im5 = axes[4].pcolormesh(X, Y, results['avg_fidelity_full_rates'], 
                            shading='auto', cmap='viridis', vmin=0, vmax=1)
    axes[4].set_title('Average Fidelity: Full vs Rates')
    axes[4].set_ylabel('Input Power (dB)')
    axes[4].set_xlabel('Target Frequency (MHz)')
    plt.colorbar(im5, ax=axes[4])
    
    # Plot 6: Average Fidelity - Full vs Redfield
    im6 = axes[5].pcolormesh(X, Y, results['avg_fidelity_full_redfield'], 
                            shading='auto', cmap='viridis', vmin=0, vmax=1)
    axes[5].set_title('Average Fidelity: Full vs Redfield')
    axes[5].set_xlabel('Target Frequency (MHz)')
    plt.colorbar(im6, ax=axes[5])
    
    # Plot 7: Target State <σ_x>
    im7 = axes[6].pcolormesh(X, Y, results['heatmapx_target_state'], 
                            shading='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes[6].set_title('Target State <σ_x>')
    axes[6].set_xlabel('Target Frequency (MHz)')
    plt.colorbar(im7, ax=axes[6])
    
    # Plot 8: Target State <σ_z>
    im8 = axes[7].pcolormesh(X, Y, results['heatmapz_target_state'], 
                            shading='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes[7].set_title('Target State <σ_z>')
    axes[7].set_xlabel('Target Frequency (MHz)')
    plt.colorbar(im8, ax=axes[7])
    
    # Add overall title with parameters
    params_text = []
    if 'rabi_freq' in results:
        params_text.append(f"Rabi: {results['rabi_freq']/(2*np.pi*1e6):.1f} MHz")
    if 'cavity_diss_rate' in results:
        params_text.append(f"κ: {results['cavity_diss_rate']/(2*np.pi*1e6):.1f} MHz")
    if 'eff_coupling' in results:
        params_text.append(f"g: {results['eff_coupling']/(2*np.pi*1e6):.1f} MHz")
    
    plt.suptitle(' | '.join(params_text), y=0.98)
    plt.tight_layout()
    
    if save_plots:
        # Create filename based on parameters
        filename_parts = []
        if 'cavity_diss_rate' in results:
            filename_parts.append(f"kappa_{results['cavity_diss_rate']/(2*np.pi*1e6):.1f}MHz")
        if 'rabi_freq' in results:
            filename_parts.append(f"omega_{results['rabi_freq']/(2*np.pi*1e6):.1f}MHz")
        if 'eff_coupling' in results:
            filename_parts.append(f"g_{results['eff_coupling']/(2*np.pi*1e6):.1f}MHz")
        
        filename = f"heatmaps_{'_'.join(filename_parts)}.png"
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        print("Plot created but not displayed (show_plot=False)")
    
    # Return the figure so caller can decide when to close it
    return fig, axes

def plot_and_close_heatmaps(results, save_plots=False, plot_dir="./heatmap_plots", show_plot=True):
    """
    Plot heatmaps and immediately close the figure to save memory
    """
    fig, axes = plot_heatmaps(results, save_plots=save_plots, plot_dir=plot_dir, show_plot=show_plot)
    
    # Close the figure to free memory
    plt.close(fig)
    
    return fig, axes

def plot_all_files_in_directory(directory, save_plots=True, plot_dir="./heatmap_plots", 
                               close_figures=True, show_plots=True):
    """
    Main function to load and plot all result files in a directory
    with proper memory management
    
    Parameters:
    - directory: directory containing result files
    - save_plots: whether to save plots to files
    - plot_dir: directory to save plots
    - close_figures: whether to close figures after plotting (saves memory)
    - show_plots: whether to display plots on screen
    """
    print(f"Processing all files in: {directory}")
    print("=" * 60)
    
    # Find all result files
    all_files = find_all_results_files(directory)
    
    all_results = {}
    
    for filepath in all_files:
        print(f"\nProcessing: {os.path.basename(filepath)}")
        print("-" * 40)
        
        try:
            # Load results
            results = read_results(filepath)
            
            # Plot with memory management
            if close_figures:
                plot_and_close_heatmaps(results, save_plots=save_plots, 
                                      plot_dir=plot_dir, show_plot=show_plots)
            else:
                plot_heatmaps(results, save_plots=save_plots, 
                            plot_dir=plot_dir, show_plot=show_plots)
            
            # Store results for potential further analysis
            filename = os.path.basename(filepath)
            all_results[filename] = results
            
            print(f"Successfully processed: {filename}")
            
        except Exception as e:
            print(f"ERROR processing {filepath}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Completed processing {len(all_results)} files")
    
    return all_results

def plot_selected_comparison(directory, save_plots=True, plot_dir="./comparison_plots", 
                           show_plot=True, close_figures=True):
    """
    Plot only the most important heatmaps for comparison
    """
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    all_files = find_all_results_files(directory)
    
    for filepath in all_files:
        print(f"Creating comparison plot for: {os.path.basename(filepath)}")
        
        try:
            results = read_results(filepath)
            
            # Create a smaller figure with just the key plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            input_power_array = results['input_power_array']
            qubit_targetz_array = results['qubit_targetz_array'] / (2 * np.pi * 1e6)
            X, Y = np.meshgrid(qubit_targetz_array, input_power_array)
            
            # Plot only the most important ones
            plots = [
                ('final_fidelity_full_redfield', 'Full vs Redfield (Final)'),
                ('avg_fidelity_full_redfield', 'Full vs Redfield (Average)'),
                ('final_fidelity_full_target', 'Full vs Target (Final)'),
                ('heatmapz_target_state', 'Target State <σ_z>')
            ]
            
            for idx, (key, title) in enumerate(plots):
                im = axes[idx].pcolormesh(X, Y, results[key], shading='auto', cmap='viridis')
                axes[idx].set_title(title)
                axes[idx].set_xlabel('Target Frequency (MHz)')
                axes[idx].set_ylabel('Input Power (dB)')
                plt.colorbar(im, ax=axes[idx])
            
            # Add parameter info
            params_text = []
            if 'rabi_freq' in results:
                params_text.append(f"Ω: {results['rabi_freq']/(2*np.pi*1e6):.1f} MHz")
            if 'cavity_diss_rate' in results:
                params_text.append(f"κ: {results['cavity_diss_rate']/(2*np.pi*1e6):.1f} MHz")
            if 'eff_coupling' in results:
                params_text.append(f"g: {results['eff_coupling']/(2*np.pi*1e6):.1f} MHz")
            
            plt.suptitle(' | '.join(params_text), y=0.98)
            plt.tight_layout()
            
            if save_plots:
                filename_parts = []
                if 'cavity_diss_rate' in results:
                    filename_parts.append(f"kappa_{results['cavity_diss_rate']/(2*np.pi*1e6):.1f}MHz")
                if 'rabi_freq' in results:
                    filename_parts.append(f"omega_{results['rabi_freq']/(2*np.pi*1e6):.1f}MHz")
                if 'eff_coupling' in results:
                    filename_parts.append(f"g_{results['eff_coupling']/(2*np.pi*1e6):.1f}MHz")
                
                filename = f"comparison_{'_'.join(filename_parts)}.png"
                filepath = os.path.join(plot_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Comparison plot saved to: {filepath}")
            
            if show_plot:
                plt.show()
            
            if close_figures:
                plt.close(fig)
                
        except Exception as e:
            print(f"ERROR processing {filepath}: {e}")
            continue

def main():
    """
    Main function to plot all files in different directories
    with configurable display options
    """
    # Define directories to process
    directories = [
        "./diss_rate_results",
        "./rabi_freq_results", 
        "./eff_coupling_results"
    ]
    
    # Configuration options
    config = {
        'save_plots': True,      # Save plots to files
        'show_plots': False,     # Display plots on screen (set to True if you want to see them)
        'close_figures': True,   # Close figures to save memory
        'plot_dir': "./heatmap_plots"  # Directory to save plots
    }
    
    all_results = {}
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\n{'='*60}")
            print(f"PROCESSING DIRECTORY: {directory}")
            print(f"{'='*60}")
            
            try:
                results = plot_all_files_in_directory(
                    directory, 
                    save_plots=config['save_plots'],
                    plot_dir=config['plot_dir'],
                    close_figures=config['close_figures'],
                    show_plots=config['show_plots']
                )
                all_results[directory] = results
            except Exception as e:
                print(f"Error processing directory {directory}: {e}")
        else:
            print(f"Directory {directory} does not exist, skipping...")
    
    return all_results

# Quick usage examples:
if __name__ == "__main__":
    # Option 1: Process all files without showing plots (for batch processing)
    all_results = main()
    
    # Option 2: Process specific directory and show plots
    # plot_all_files_in_directory("./diss_rate_results", save_plots=True, show_plots=True)
    
    # Option 3: Process single file and show plot
    # results = read_results("./diss_rate_results/your_file.npz")
    # plot_heatmaps(results, save_plots=True, show_plot=True)
    
    # Option 4: Just save plots without displaying (for remote/server use)
    # plot_all_files_in_directory("./diss_rate_results", save_plots=True, show_plots=False, close_figures=True)