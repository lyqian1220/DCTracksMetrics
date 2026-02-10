import numpy as np
import matplotlib.pyplot as plt
import os

from src.match.matcher import Match
from src.match import metrics
from src.performance.initializer import cache_results, load_evt

def get_hit_data(mc_dict, good_track_dict, mdc_track_dict):
    """
    Extract hit-related data (pT, cos, phi, efficiency, purity) from track dictionaries
    """
    pt_values = []       
    cos_values = []      
    phi_values = []      
    eff_values = []      
    purity_values = []   

    for i, (evt_good_dict, evt_mdc_dict, evt_mc_dict) in enumerate(zip(good_track_dict, mdc_track_dict, mc_dict)):
        if not evt_good_dict:  
            continue 
        
        for trkid, track in evt_good_dict.items():
            rec_dict = track.get('Rec', {})          
            mc_idx = next(iter(rec_dict.keys()))     
            mdc_q = evt_mdc_dict[trkid]["Mdc"]["charge"]  
            
            if mc_idx not in evt_mc_dict:
                continue
            mc_q = evt_mc_dict[mc_idx]["Mc"]["charge"]  
            
            # Only keep tracks with correct charge
            if mdc_q != mc_q:
                continue
            
            # Extract MC track kinematic parameters and reconstruction metrics
            pt_values.append(evt_mc_dict[mc_idx]["Mc"]["pT"])
            cos_values.append(evt_mc_dict[mc_idx]["Mc"]["cos"])
            phi_values.append(evt_mc_dict[mc_idx]["Mc"]["phi"])
            eff_values.append(rec_dict[mc_idx]["eff"])
            purity_values.append(rec_dict[mc_idx]["purity"])

    return pt_values, cos_values, phi_values, eff_values, purity_values

def get_hit_violin_data(
    common_mc_dict, 
    good_track_dict_trad, good_track_dict_ml, 
    mdc_track_dict_trad, mdc_track_dict_ml,
    ):
    """
    Prepare violin plot data for comparing baseline and ML-based method
    """
    # Extract data for ML finder
    pt_values_ml, cos_values_ml, phi_values_ml, eff_values_ml, purity_values_ml = get_hit_data(
        mc_dict=common_mc_dict,
        good_track_dict=good_track_dict_ml,
        mdc_track_dict=mdc_track_dict_ml,
    )

    # Extract data for baseline finder
    pt_values_trad, cos_values_trad, phi_values_trad, eff_values_trad, purity_values_trad = get_hit_data(
        mc_dict=common_mc_dict,
        good_track_dict=good_track_dict_trad,
        mdc_track_dict=mdc_track_dict_trad,
    )

    hit_violin_data = {'eff_pt': [], 'eff_cos': [], 'purity_pt': [], 'purity_cos': [], 'eff_phi': [], 'purity_phi': []}
    hit_violin_labels = {'ml': "GNN Finder", 'trad': "Baseline Finder"}  
    hit_violin_colors = {'ml': '#e18060', 'trad': '#2d4069'}   
    
    hit_violin_configs = [
        ('eff_pt', 'ml', eff_values_ml, pt_values_ml),
        ('eff_pt', 'trad', eff_values_trad, pt_values_trad),
        ('eff_cos', 'ml', eff_values_ml, cos_values_ml),
        ('eff_cos', 'trad', eff_values_trad, cos_values_trad),
        ('eff_phi', 'ml', eff_values_ml, phi_values_ml),
        ('eff_phi', 'trad', eff_values_trad, phi_values_trad),
        ('purity_pt', 'ml', purity_values_ml, pt_values_ml),
        ('purity_pt', 'trad', purity_values_trad, pt_values_trad),
        ('purity_cos', 'ml', purity_values_ml, cos_values_ml),
        ('purity_cos', 'trad', purity_values_trad, cos_values_trad),
        ('purity_phi', 'ml', purity_values_ml, phi_values_ml),
        ('purity_phi', 'trad', purity_values_trad, phi_values_trad),
    ]

    # Populate violin plot data with labels, values, binning variables and colors
    for metric_key, data_type, values, bin_values in hit_violin_configs:
        label = hit_violin_labels[data_type]
        color = hit_violin_colors[data_type]
        hit_violin_data[metric_key].append((label, values, bin_values, color))

    return hit_violin_data

def plot_violin(data, bins, widths, xlabel, ylabel, ylim, yticks, data_type):
    """
    Generate violin plots for track finding performance metrics binned by kinematic variables
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    n_bins = len(bins) - 1                 
    bin_centers = (bins[:-1] + bins[1:]) / 2  
    positions = bin_centers     

    bin_counts = []   
    all_finder_data = []

    for j, (label, values, bin_values, color) in enumerate(data):
        counts_per_bin = []
        current_finder_all_data = [] 
        
        for i in range(n_bins):
            mask = (np.array(bin_values) >= bins[i]) & (np.array(bin_values) < bins[i+1])
            count = np.sum(mask)   
            counts_per_bin.append(count)
            
            current_bin_data = [values[k] for k, m in enumerate(mask) if m]
            current_finder_all_data.extend(current_bin_data)
        
        bin_counts.append(counts_per_bin)
        all_finder_data.append(current_finder_all_data)

    for i in range(n_bins):
        bin_center_str = f"{bin_centers[i]:.2f}"
        baseline_count = bin_counts[0][i] if len(bin_counts) >= 1 else 0
        gnn_count = bin_counts[1][i] if len(bin_counts) >= 2 else 0
        total_count = baseline_count + gnn_count
        print(f"{bin_center_str:<10} {baseline_count:<15} {gnn_count:<15} {total_count:<10}")

    for j, (label, _, _, _) in enumerate(data):
        finder_all_data = all_finder_data[j]
        
        valid_data = np.array([x for x in finder_all_data if not np.isnan(x)])
        n_valid = len(valid_data)
        
        if n_valid > 0:
            global_mean = np.mean(valid_data)  
            binomial_err = np.sqrt(global_mean * (1 - global_mean) / n_valid) if n_valid > 1 else 0.0
            
            print(f"{label:<12} {global_mean*100:>6.2f}% ± {binomial_err*100:>5.2f}% "
                f"   (N = {n_valid:,})")
            
        else:
            print(f"{label:<12} No valid data to calculate")
    
    handles = []
    # Generate violin plots for each finder
    for j, (label, values, bin_values, color) in enumerate(data):
        binned_data = []

        for i in range(n_bins):
            mask = (np.array(bin_values) >= bins[i]) & (np.array(bin_values) < bins[i+1])
            binned = [values[k] for k, m in enumerate(mask) if m]
            binned_data.append(binned if binned else [np.nan])  # Use NaN for empty bins
        
        # Plot violin 
        parts = ax.violinplot(
            binned_data,
            positions=positions + (-1)**j * 0.001,  
            points=100,                              
            widths=widths,                           
            showmeans=True,                          # Show mean value marker
            showextrema=False,                       # Hide min/max markers
            showmedians=True,                        # Show median value marker
            quantiles=None,
            bw_method=0.5,                           # Bandwidth scaling for kernel density estimation
            side='high' if j == 0 else 'low'         # Draw violins on different sides for clarity
        )
    
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(1)
        
        parts['cmedians'].set_color('gray') 
        parts['cmedians'].set_linestyle('--') 
        parts['cmedians'].set_linewidth(1.)

        parts['cmeans'].set_color('gray') 
        parts['cmeans'].set_linestyle('--')  
        parts['cmeans'].set_linewidth(1.)

        handles.append(parts['bodies'][0])
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{c:.2f}' for c in bin_centers])  
    ax.set_xlabel(xlabel, fontsize=13)                     
    ax.set_ylabel(ylabel, fontsize=13)                     
    ax.set_ylim(ylim)                                      
    ax.set_yticks(yticks)                                  
    ax.tick_params(axis='both', labelsize=11)              

    for boundary in bins:  
        ax.vlines(
            x=boundary,
            color='gray',
            linestyle='--',
            linewidth=1.3,
            alpha=0.3,
            ymin=0.0,
            ymax=1.0
        )
    
    ax.set_facecolor('white')
    ax.legend(handles, [d[0] for d in data], loc='lower right', frameon=True, fontsize=12)
    
    ax.text(0.90, 0.97, f'{data_type}',
            fontsize=13,
            ha='right', va='top',
            transform=ax.transAxes,
            color='black')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    plt.tight_layout()  
    plt.close()         

    return fig