import matplotlib.pyplot as plt
import numpy as np

from src.performance import config
from src.performance.initializer import cache_results
from src.performance import data_loader

DEFAULT_CACHE_PATH = "./cache"  # Default path for caching intermediate results

def get_mc_truth_data(mc_dict, param_name):
    """
    Extract MC truth values for a specific parameter from MC dictionary
    """
    param_truth = []
    for evt_mc_dict in mc_dict:
        for track_index, track_info in evt_mc_dict.items():
            param_truth.append(track_info["Mc"][param_name])

    param_truth_np = np.array(param_truth)
    return param_truth_np

def match_good_charge_track(evt_label, common_mc_dict, track_dict, good_track_dict, track_type, param_name,  cache_path):
    """
    Match good tracks with correct charge to MC truth and calculate kinematic parameters difference
    """
    param_data = [] 
    param_diff = [] 

    for i, (evt_good_dict, evt_type_dict, evt_mc_dict) in enumerate(zip(good_track_dict, track_dict, common_mc_dict)):
        if not evt_good_dict or not evt_mc_dict:
            continue
        
        for trkid, track in evt_good_dict.items():
            track_rec_dict = track.get('Rec', {})
            trackIndex = next(iter(track_rec_dict.keys())) 
            
            # Get charge from reconstructed track
            type_q = evt_type_dict[trkid][track_type]["charge"]

            if trackIndex not in evt_mc_dict:
                continue
            # Get charge from MC truth
            mc_q = evt_mc_dict[trackIndex]["Mc"]["charge"]
            
            # Kalman track requires additional status check (status 0 = track passed the fitting)
            if track_type == "Kal":
                status = track.get('status', {})
                if type_q == mc_q and status == 0:
                    param_data.append(evt_type_dict[trkid][track_type][param_name])
                    param_diff.append(evt_type_dict[trkid][track_type][param_name] - evt_mc_dict[trackIndex]["Mc"][param_name])
            # MDC track requires charge match
            else: 
                if type_q == mc_q:
                    param_data.append(evt_type_dict[trkid][track_type][param_name])
                    param_diff.append(evt_type_dict[trkid][track_type][param_name] - evt_mc_dict[trackIndex]["Mc"][param_name])
    
    param_data = np.array(param_data)
    param_diff = np.array(param_diff)

    return (param_data, param_diff)


def match_common_good_charge_track(
    common_mc_dict, 
    track_dict_trad,  
    track_dict_ml,      
    good_track_dict_trad,  
    good_track_dict_ml,  
    track_type, 
    param_name,  
):
    """
    Match common good tracks (present in both baseline and ML methods) with correct charge to MC truth
    Calculate normalized parameter differences (rec - MC)/MC 
    """
    trad_param_data = []
    ml_param_data = []
    trad_param_diff = []  # (baseline rec - MC)/MC
    ml_param_diff = []    # (ML rec - MC)/MC
    mc_data = []          # MC truth values for target parameter

    mc_pt_data = []       # MC truth pT values (transverse momentum)

    for i, (evt_good_trad, evt_good_ml, evt_track_trad, evt_track_ml, evt_mc_dict) in enumerate(zip(
        good_track_dict_trad, good_track_dict_ml, track_dict_trad, track_dict_ml, common_mc_dict)):

        if not evt_good_trad or not evt_good_ml or not evt_mc_dict:
            continue

        # Create track index to track ID mapping for baseline method
        trad_trackIndex_to_trkid = {}
        for trkid, good_track in evt_good_trad.items():
            rec_dict = good_track.get('Rec', {})
            trackIndex = next(iter(rec_dict.keys()))  
            trad_trackIndex_to_trkid[trackIndex] = trkid  

        # Create track index to track ID mapping for ML method
        ml_trackIndex_to_trkid = {}
        for trkid, good_track in evt_good_ml.items():
            rec_dict = good_track.get('Rec', {})
            trackIndex = next(iter(rec_dict.keys()))  
            ml_trackIndex_to_trkid[trackIndex] = trkid  

        # Find common track indices (present in both baseline data, ML data and MC data)
        trad_valid_trackIndices = set(trad_trackIndex_to_trkid.keys()) & set(evt_mc_dict.keys())
        ml_valid_trackIndices = set(ml_trackIndex_to_trkid.keys()) & set(evt_mc_dict.keys())
        common_trackIndices = trad_valid_trackIndices & ml_valid_trackIndices 

        if not common_trackIndices:
            continue  

        # Process each common track index
        for trackIndex in common_trackIndices:
            # Get track IDs for baseline and ML methods
            trad_trkid = trad_trackIndex_to_trkid[trackIndex]
            ml_trkid = ml_trackIndex_to_trkid[trackIndex]
            
            mc_charge = evt_mc_dict[trackIndex]["Mc"]["charge"]  # MC truth charge
            trad_charge = evt_track_trad[trad_trkid][track_type]["charge"]  # Baseline rec charge
            ml_charge = evt_track_ml[ml_trkid][track_type]["charge"]      # ML rec charge

            if track_type == "Kal":
                trad_status = evt_good_trad[trad_trkid].get('status', -1)
                ml_status = evt_good_ml[ml_trkid].get('status', -1)
                # Skip if charge mismatch or bad status (status 0 = good)
                if (trad_charge != mc_charge) or (ml_charge != mc_charge) or (trad_status != 0) or (ml_status != 0):
                    continue
            else: 
                if (trad_charge != mc_charge) or (ml_charge != mc_charge):
                    continue

            # Extract MC truth values
            mc_param = evt_mc_dict[trackIndex]["Mc"][param_name]  # Target parameter MC truth
            mc_pt = evt_mc_dict[trackIndex]["Mc"]["pT"]           # Transverse momentum MC truth

            mc_data.append(mc_param)
            mc_pt_data.append(mc_pt)
            
            # Calculate baseline method parameter and normalized difference
            trad_param = evt_track_trad[trad_trkid][track_type][param_name]
            trad_param_diff.append((trad_param - mc_param) / mc_param if mc_param else 0)  # Avoid division by zero
            trad_param_data.append(trad_param)
            
            # Calculate ML method parameter and normalized difference
            ml_param = evt_track_ml[ml_trkid][track_type][param_name]
            ml_param_diff.append((ml_param - mc_param) / mc_param if mc_param else 0)  # Avoid division by zero
            ml_param_data.append(ml_param)

    return trad_param_data, ml_param_data, trad_param_diff, ml_param_diff, mc_pt_data, mc_data



def plot_coverage(ax, truth_data, diff_data, bins, track_type, linestyle, label, color=None, marker=None):
    """
    Calculate and plot resolution
    """

    truth_data = np.array(truth_data)
    diff_data = np.array(diff_data)
    bins = np.array(bins)
    
    if len(truth_data) != len(diff_data):
        raise ValueError(f"truth_data and diff_data length mismatch: {len(truth_data)} vs {len(diff_data)}")
    
    resolution_means = []  
    bin_centers = []  
    entries_per_bin = []  
    
    x_errors = [] 
    y_errors_std = []  
    y_errors_mad = []  
    
    for i in range(len(bins) - 1):
        bin_mask = (truth_data >= bins[i]) & (truth_data < bins[i + 1])
        
        bin_truth = truth_data[bin_mask]
        bin_diff = diff_data[bin_mask]
        
        entries_count = len(bin_truth)
        entries_per_bin.append(entries_count)
        
        bin_width = (bins[i + 1] - bins[i]) / 2
        x_errors.append(bin_width)
        
        if entries_count > 10: 
            # Calculate median of differences (central value)
            median = np.median(bin_diff)
            # Absolute deviation from median
            abs_deviation = np.abs(bin_diff - median)
            # Resolution = 68th percentile (1 sigma equivalent for non-Gaussian)
            resolution = np.percentile(abs_deviation, 68)
            
            resolution_means.append(resolution)
            bin_centers.append((bins[i] + bins[i + 1]) / 2) 
            
            # Calculate statistical errors
            y_std = np.std(bin_diff) 
            y_errors_std.append(y_std)
            y_mad = np.median(np.abs(bin_diff - np.median(bin_diff)))
            y_errors_mad.append(y_mad)
            
        else:
            resolution_means.append(np.nan)
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            y_errors_std.append(np.nan)
            y_errors_mad.append(np.nan)
    
    valid_mask = ~np.isnan(resolution_means)
    bin_centers_valid = np.array(bin_centers)[valid_mask]
    resolution_means_valid = np.array(resolution_means)[valid_mask]
    x_errors_valid = np.array(x_errors)[valid_mask]
    y_errors_std_valid = np.array(y_errors_std)[valid_mask]
    y_errors_mad_valid = np.array(y_errors_mad)[valid_mask]
    
    # Plot resolution curve
    ax.plot(bin_centers_valid, resolution_means_valid,
            linestyle=linestyle, marker=marker, color=color, 
            label=label, linewidth=2)
    
    return {
        'label': label,
        'entries_count': len(diff_data),
        'resolution_values': resolution_means_valid,
        'truth_centers': bin_centers_valid,
        'line_color': color,
        'entries_per_bin': entries_per_bin,
        'x_errors': x_errors_valid,
        'y_errors_std': y_errors_std_valid,
        'y_errors_mad': y_errors_mad_valid
    }
    


def plot_resolution_dual_mdc_kal(
    data_type,
    common_mc_dict,         
    track_dict_kal_trad, track_dict_kal_ml,
    good_track_dict_kal_trad, good_track_dict_kal_ml,
    track_dict_mdc_trad, track_dict_mdc_ml,
    good_track_dict_mdc_trad, good_track_dict_mdc_ml,
    param_name="dr",
    cache_path=DEFAULT_CACHE_PATH
):
    """
    Main function to plot resolution comparison between MDC and Kalman tracks for baseline and ML methods
    """

    colors = {
        "Kal Baseline":  (45/255, 64/255, 105/255),   
        "Kal GNN":       (225/255, 128/255, 96/255),  
        "MDC Baseline":  (45/255, 64/255, 105/255),   
        "MDC GNN":       (225/255, 128/255, 96/255),  
    }
    
    # Get matched data for Kalman tracks (baseline vs ML)
    kal_trad_data, kal_ml_data, kal_trad_diff, kal_ml_diff, mc_pt_data_kal, mc_data_kal = match_common_good_charge_track(
        common_mc_dict,
        track_dict_kal_trad, track_dict_kal_ml,
        good_track_dict_kal_trad, good_track_dict_kal_ml,
        track_type="Kal", param_name=param_name
    )

    # Get matched data for MDC tracks (baseline vs ML)
    mdc_trad_data, mdc_ml_data, mdc_trad_diff, mdc_ml_diff, mc_pt_data_mdc, mc_data_mdc = match_common_good_charge_track(
        common_mc_dict,
        track_dict_mdc_trad, track_dict_mdc_ml,
        good_track_dict_mdc_trad, good_track_dict_mdc_ml,
        track_type="Mdc", param_name=param_name
    )

    config_rk  = config.resolution_config(param_name)       # Kalman resolution config
    config_mdc = config.resolution_config_MDC(param_name)   # MDC resolution config
    config_pt = config.resolution_config("pT")              # pT binning config

    param_bins  = np.linspace(config_rk["range_min"],  config_rk["range_max"],  config_rk["num_bins"])
    diff_bins   = np.linspace(config_rk["diff_range_min"], config_rk["diff_range_max"], config_rk["diff_num_bins"])
    c_param_bins = np.linspace(config_rk["range_min"], config_rk["range_max"], config_rk["c_num_bins"])
    pt_bins = np.linspace(config_pt["range_min"], config_pt["range_max"], config_pt["c_num_bins"])

    # --------------------------
    # Plot 1: Parameter Distribution Histogram
    # --------------------------
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6)) 
    fig1.subplots_adjust(
        left=0.12,  
        right=0.88,  
        top=0.85,  
        bottom=0.15)

    n_truth, _ = np.histogram(mc_data_mdc, bins=param_bins)
    n_kal_trad, _ = np.histogram(kal_trad_data, bins=param_bins)
    n_kal_ml,   _ = np.histogram(kal_ml_data,   bins=param_bins)
    n_mdc_trad, _ = np.histogram(mdc_trad_data, bins=param_bins)
    n_mdc_ml,   _ = np.histogram(mdc_ml_data,   bins=param_bins)
    
    global_max = max(n_truth.max(), n_kal_trad.max(), n_kal_ml.max(),
                     n_mdc_trad.max(), n_mdc_ml.max()) * 1.6

    ax1.hist(mc_data_mdc,    bins=param_bins, histtype='step', lw=1.5, label="MC Truth", color="red", alpha=1)
    ax1.hist(mdc_trad_data,  bins=param_bins, histtype='step', lw=1.5, linestyle='-', label="Baseline Finder", color=colors["MDC Baseline"])
    ax1.hist(mdc_ml_data,    bins=param_bins, histtype='step', lw=1.5, linestyle='-', label="GNN Finder",     color=colors["MDC GNN"])
    ax1.hist(kal_trad_data,  bins=param_bins, histtype='step', lw=1.8, linestyle='--', label="Baseline Fitter", color=colors["Kal Baseline"])
    ax1.hist(kal_ml_data,    bins=param_bins, histtype='step', lw=1.8, linestyle='--', label="GNN Fitter",     color=colors["Kal GNN"])

    ax1.set_xlabel(f"${param_name_latex(param_name)}$ " + config.resolution_config(param_name)["xlabel"], fontsize=20)
    ax1.set_ylabel("Number of Tracks", fontsize=20)
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))  # Scientific notation for y-axis
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.set_ylim(0, global_max)
    ax1.legend(ncol=1, loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=16, facecolor='none')
    ax1.tick_params(axis='both', which='major', labelsize=20, length=8, width=2)
    ax1.text(
        0.05, 0.97, f'{data_type}',
        fontsize=20, ha='left', va='top',
        transform=ax1.transAxes, color='black'
    )
    
    # --------------------------
    # Plot 2: Resolution vs pT
    # --------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7.5))
    plot_coverage(ax2, mdc_trad_data, mdc_trad_diff, pt_bins, "Mdc", '-',  "Baseline Finder", color=colors["MDC Baseline"])
    plot_coverage(ax2, mdc_ml_data,   mdc_ml_diff,   pt_bins, "Mdc", '-',  "GNN Finder",      color=colors["MDC GNN"])
    plot_coverage(ax2, kal_trad_data, kal_trad_diff, pt_bins, "Kal", '--', "Baseline Fitter", color=colors["Kal Baseline"])
    plot_coverage(ax2, kal_ml_data,   kal_ml_diff,   pt_bins, "Kal", '--', "GNN Fitter",      color=colors["Kal GNN"])

    ax2.set_xlabel(f"${param_name_latex('pT')}^\mathrm{{MC}}\ [GeV/c]$", fontsize=30)
    ax2.set_ylabel(f'Resolution$({param_name_latex(param_name)}^{{pred}} - {param_name_latex(param_name)}^{{MC}}) / {param_name_latex(param_name)}^{{MC}}$ ', fontsize=30)
    ax2.tick_params(axis='both', which='major', labelsize=30, length=8, width=2)
    ax2.legend(ncol=1, loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=25, facecolor='none')

    ax2.text(
        0.05, 0.97, f'{data_type}',
        fontsize=30, ha='left', va='top',
        transform=ax2.transAxes, color='black'
    )

    ax2.set_yscale('log')
    ax2.set_ylim(*config.resolution_config(param_name).get("resolution_ylim", [1e-3, 1e2]))
    
    
    return fig1, fig2


def param_name_latex(name):
    
    mapping = {
        "dr": r"d_r",          
        "dz": r"d_z",        
        "phi0": r"\phi_0",     
        "tanl": r"\tan\lambda",
        "kappa": r"\kappa",    
        "pT": r"p_T",          
        "pz": r"p_z",          
        "px": r"p_x",         
        "py": r"p_y",          
        "p": r"p"              
    }

    return mapping.get(name, name)  