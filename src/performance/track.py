import matplotlib.pyplot as plt
import numpy as np

from src.match import metrics
from src.match.matcher import Match
from src.performance.initializer import get_cache_key, cache_results


def extract_common_mc_info(mc_dict_trad, mc_dict_ml):
    """
    Extract common Monte Carlo (MC) track information between baseline and ML methods
    """
    common_mc_dict = [] 
    
    for trad_evt, ml_evt in zip(mc_dict_trad, mc_dict_ml):
        current_common = {}
        
        if not trad_evt or not ml_evt:
            common_mc_dict.append(current_common)
            continue
        
        # Find intersection of track indices between two methods
        common_track_indices = set(trad_evt.keys()) & set(ml_evt.keys())
        if not common_track_indices:
            common_mc_dict.append(current_common)
            continue
        
        # Extract common track information from MC dict
        for track_idx in common_track_indices:
            current_common[track_idx] = trad_evt[track_idx]
        
        common_mc_dict.append(current_common)

    return common_mc_dict



def compute_denominator(common_mc_info_dict, x_type, xbins):
    """
    Compute denominator (total number of MC tracks) for efficiency calculation per bin
    Args:
        common_mc_info_dict: Common MC track information 
        x_type: Variable type for binning (e.g., 'pT', 'cos')
        xbins: Bin edges for the x variable
    Returns:
        denominator: Array of total MC tracks per bin
    """
    denominator = np.zeros(len(xbins) - 1)
    for evt_common in common_mc_info_dict:
        if not evt_common:
            continue
        
        # Count MC tracks in each bin based on x_type variable
        for track_info in evt_common.values():
            xvar = track_info["Mc"][x_type]
            bin_idx = np.searchsorted(xbins, xvar, side='right') - 1
            if 0 <= bin_idx < len(denominator):
                denominator[bin_idx] += 1
    
    return denominator

def compute_numerator(
    common_mc_info_dict,
    track_type, x_type, 
    track_dict, track_mdc_dict, track_kal_dict, 
    xbins, is_charge=False, is_charge_wrong=False
    ):
    """
    Compute numerator (number of reconstructed tracks) for efficiency calculation per bin
    Supports different track types (Good/Fake/Clone/Charge/ChargeWrong)
    Args:
        common_mc_info_dict: Common MC track information across methods
        track_type: Type of track to calculate (Good/Fake/Clone/Charge/ChargeWrong)
        x_type: Variable type for binning (e.g., 'pT', 'cos')
        track_dict: Reconstructed track dictionary
        track_mdc_dict: MDC (Main Drift Chamber) track information
        track_kal_dict: Kalman filter track information
        xbins: Bin edges for the x variable
        is_charge: Whether to calculate charge matching efficiency
        is_charge_wrong: Whether to calculate wrong charge rate
    Returns:
        finder_numerator: Numerator for finder efficiency
        fitter_numerator: Numerator for fitter efficiency (only for status=0 tracks)
    """
    # Initialize numerator arrays for finder and fitter
    finder_numerator = np.zeros(len(xbins) - 1)
    fitter_numerator = np.zeros(len(xbins) - 1)
    
    # Initialize charge-related numerators if needed
    if is_charge:
        charge_finder_numerator = np.zeros(len(xbins) - 1)
        charge_fitter_numerator = np.zeros(len(xbins) - 1)

    if is_charge_wrong:
        wrong_charge_finder_numerator = np.zeros(len(xbins) - 1)
        wrong_charge_fitter_numerator = np.zeros(len(xbins) - 1)
    
    # Iterate over events (track data + MC data + MDC data + Kalman data)
    for i, (evt_track_dict, evt_mc_dict, evt_mdc_dict, evt_kal_dict) in enumerate(zip(
        track_dict, common_mc_info_dict, track_mdc_dict, track_kal_dict
    )):  
        if not evt_mc_dict or not evt_track_dict:
            continue

        # Iterate over reconstructed tracks in current event
        for trkid, track in evt_track_dict.items():
            rec_dict = track.get('Rec', {})
            
            # Handle Good/Charge/ChargeWrong track types (MC-matched tracks)
            if track_type == "Good" or track_type == "Charge" or track_type == "ChargeWrong":
                mc_idx = next(iter(rec_dict.keys()))  # Get matched MC track index
                if mc_idx not in evt_mc_dict:
                    continue
                xvar = evt_mc_dict[mc_idx]["Mc"][x_type]  # Get MC variable for binning
                status = track.get('status', {})  # Track fit status
            
            # Handle Fake/Clone track types (no MC match)
            elif track_type == "Fake" or track_type == "Clone":
                if trkid in evt_mdc_dict:
                    xvar = evt_mdc_dict[trkid]["Mdc"][x_type]  # Use MDC variable for binning
                    status = evt_kal_dict[trkid]["Kal"]["status"]  # Kalman fit status
            
            # Determine bin index for current track's x variable
            bin_idx = np.searchsorted(xbins, xvar, side='left') 
            if bin_idx == 0:
                bin_idx = 0
            elif bin_idx == len(xbins):
                bin_idx = len(finder_numerator) - 1
            else:
                bin_idx = bin_idx - 1
            
            # Skip if bin index is out of valid range
            if not (0 <= bin_idx <= len(finder_numerator) - 1): 
                continue

            # Calculate charge matching numerators
            if is_charge:
                mdc_q = evt_mdc_dict[trkid]["Mdc"]["charge"]  # Charge from MDC finder
                kal_q = evt_kal_dict[trkid]["Kal"]["charge"]  # Charge from Kalman fitter
                mc_q = evt_mc_dict[mc_idx]["Mc"]["charge"]    # True MC charge
                
                # Count correct charge from finder
                if mdc_q == mc_q:
                    charge_finder_numerator[bin_idx] += 1
                # Count correct charge from fitter (only for successful fits)
                if kal_q == mc_q and status == 0:
                    charge_fitter_numerator[bin_idx] += 1
            
            # Calculate wrong charge rate numerators
            elif is_charge_wrong:
                mdc_q = evt_mdc_dict[trkid]["Mdc"]["charge"]
                kal_q = evt_kal_dict[trkid]["Kal"]["charge"]
                mc_q = evt_mc_dict[mc_idx]["Mc"]["charge"]
                
                # Count wrong charge from finder
                if mdc_q != mc_q:
                    wrong_charge_finder_numerator[bin_idx] += 1
                # Count wrong charge from fitter (only for successful fits)
                if kal_q != mc_q and status == 0:
                    wrong_charge_fitter_numerator[bin_idx] += 1
            
            # Standard efficiency numerators (Good/Fake/Clone tracks)
            else:
                finder_numerator[bin_idx] += 1  # Count all found tracks
                if status == 0:  # Count only successfully fitted tracks (status=0 means success)
                    fitter_numerator[bin_idx] += 1

    # Override numerators with charge-related values if needed
    if is_charge:
        finder_numerator = charge_finder_numerator
        fitter_numerator = charge_fitter_numerator
    if is_charge_wrong:
        finder_numerator = wrong_charge_finder_numerator
        fitter_numerator = wrong_charge_fitter_numerator
    
    return finder_numerator, fitter_numerator

def compute_efficiency(numerator, denominator):
    """
    Calculate efficiency and its statistical error 
    Args:
        numerator: Number of successful events (numerator for efficiency calculation)
        denominator: Total number of events (denominator for efficiency calculation)
    Returns:
        eff: Efficiency array per bin
        err: Error array per bin
        total_eff: Overall efficiency (sum of numerator / sum of denominator)
        total_err: Overall error for total efficiency
    """

    eff = np.divide(
        numerator, 
        denominator, 
        out=np.zeros_like(numerator, dtype=float), 
        where=denominator != 0
    )
    
    mask = denominator > 0
    err = np.zeros_like(eff)
    err[mask] = np.sqrt(eff[mask] * (1 - eff[mask]) / denominator[mask])
    
    # Calculate total efficiency and error across all bins
    total_num = np.sum(numerator)
    total_den = np.sum(denominator)
    total_eff = total_num / total_den if total_den > 0 else 0.0
    total_err = np.sqrt(total_eff * (1 - total_eff) / total_den) if total_den > 0 else 0.0
    print(f"Total numerator: {total_num}, Total denominator: {total_den}, Total efficiency: {total_eff} ± {total_err}")
    return eff, err, total_eff, total_err

def plot_configs(
    ax, data_type, bin_centers, bin_widths, xlabel, ylim,
    eff_trad, err_trad, 
    eff_ml, err_ml,
    eff_fit_trad, err_fit_trad,
    eff_fit_ml, err_fit_ml,
    track_type, is_charge=False, is_charge_wrong=False,
    plot_finder=True, 
    plot_fitter=True  
):  
    """
    Configure plot aesthetics and draw efficiency error bars for traditional and ML methods
    Args:
        ax: Matplotlib axis object
        data_type: Type of data (e.g., 'Simulation', 'Data')
        bin_centers: Center values of each bin
        bin_widths: Width of each bin (for x error bars)
        xlabel: Label for x-axis
        ylim: Limits for y-axis
        eff_trad: Traditional method finder efficiency
        err_trad: Error of traditional finder efficiency
        eff_ml: ML method finder efficiency
        err_ml: Error of ML finder efficiency
        eff_fit_trad: Traditional method fitter efficiency
        err_fit_trad: Error of traditional fitter efficiency
        eff_fit_ml: ML method fitter efficiency
        err_fit_ml: Error of ML fitter efficiency
        track_type: Type of track (Good/Fake/Clone/Charge/ChargeWrong)
        is_charge: Whether plotting charge matching efficiency
        is_charge_wrong: Whether plotting wrong charge rate
        plot_finder: Whether to plot finder efficiency
        plot_fitter: Whether to plot fitter efficiency
    """
    # Plot traditional finder efficiency (solid line with triangle markers)
    if plot_finder:
        errorbar_container1 = ax.errorbar(bin_centers, eff_trad, yerr=err_trad, xerr=bin_widths,
                                          fmt='-^', capsize=3, color=(45/255, 64/255, 105/255), markersize=2, linewidth=1,  elinewidth=1,)
        line1, cap1, bar1 = errorbar_container1
    
    # Plot traditional fitter efficiency (dotted line with circle markers)
    if plot_fitter:
        errorbar_container2 = ax.errorbar(bin_centers, eff_fit_trad, yerr=err_fit_trad, xerr=bin_widths,
                                          fmt=':o', capsize=3, color=(45/255, 64/255, 105/255), markersize=2, alpha=1, linewidth=1,  elinewidth=1,)
        line2, cap2, bar2 = errorbar_container2

    # Plot ML finder efficiency (dashed line with triangle markers)
    if plot_finder:
        errorbar_container3 = ax.errorbar(bin_centers, eff_ml, yerr=err_ml, xerr=bin_widths,
                                          fmt=':^', capsize=3, color=(225/255, 128/255, 96/255), markersize=2, alpha=1, linewidth=1,  elinewidth=1, dashes=(1,1))
        line3, cap3, bar3 = errorbar_container3
    
    # Plot ML fitter efficiency (dashed line with circle markers)
    if plot_fitter:
        errorbar_container4 = ax.errorbar(bin_centers, eff_fit_ml, yerr=err_fit_ml, xerr=bin_widths,
                                          fmt=':o', capsize=3, color=(225/255, 128/255, 96/255), markersize=2, alpha=1, linewidth=1,  elinewidth=1, dashes=(1,1))
        line4, cap4, bar4 = errorbar_container4
    
    # Set axis labels and limits
    ax.set_xlabel(xlabel)
    if track_type == "Good":
        ylabel = 'Track Efficiency'
        loc = 'lower right'
    elif track_type == "Charge":
        ylabel = f'Track {track_type} Efficiency'
        loc = 'lower right'
    elif track_type == "ChargeWrong":
        ylabel = f'Wrong Charge Rate'
        loc = 'lower right'
    elif track_type == "Fake" or track_type == "Clone":
        ylabel = f'{track_type} Rate'
        loc = 'lower right'
    else:
        ylabel = 'Efficiency'
        loc = 'lower right'
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.0) 

    ax.text(
        0.9, 0.97, f'{data_type}',
        fontsize=10, ha='right', va='top',
        transform=ax.transAxes, color='black'
    )

    handles = []
    labels = []

    if plot_finder:
        if line3 is not None:
            handles.append((line3, cap3[0], bar3[0]))
            labels.append('GNN Finder')
        if line1 is not None:
            handles.append((line1, cap1[0], bar1[0]))
            labels.append('Baseline Finder')
    
    if plot_fitter:
        if line4 is not None and eff_fit_ml is not None:
            handles.append((line4, cap4[0], bar4[0]))
            labels.append('GNN Fitter')
        if line2 is not None and eff_fit_trad is not None:
            handles.append((line2, cap2[0], bar2[0]))
            labels.append('Baseline Fitter')
    
    if handles:
        ax.legend(
            handles=handles, labels=labels,
            loc=loc, ncol=2, frameon=True,
            fontsize=7,
        )


def plot_efficiency(
    common_mc_info_dict,
    data_type,
    track_type, 
    is_charge, is_charge_wrong,
    track_dict_trad, track_dict_ml, 
    track_mdc_dict_trad, track_mdc_dict_ml,
    track_kal_dict_trad, track_kal_dict_ml,
    x_type, xbins, xlabel, ylim, cache_path
    ):
    """
    Main function to calculate and plot track efficiency/rate for traditional and ML methods
    Supports different track types (Good/Fake/Clone) and charge-related metrics
    Args:
        common_mc_info_dict: Common MC track information across methods
        data_type: Type of data (e.g., 'Simulation', 'Data')
        track_type: Type of track to plot (Good/Fake/Clone/Charge/ChargeWrong)
        is_charge: Whether to calculate charge matching efficiency
        is_charge_wrong: Whether to calculate wrong charge rate
        track_dict_trad: Reconstructed tracks from traditional method
        track_dict_ml: Reconstructed tracks from ML method
        track_mdc_dict_trad: MDC info for traditional tracks
        track_mdc_dict_ml: MDC info for ML tracks
        track_kal_dict_trad: Kalman info for traditional tracks
        track_kal_dict_ml: Kalman info for ML tracks
        x_type: Variable for x-axis binning (e.g., 'pT', 'cos(theta)')
        xbins: Bin edges for x-axis variable
        xlabel: Label for x-axis
        ylim: Y-axis limits for the plot
        cache_path: Path to cache calculated results (to avoid recomputation)
    Returns:
        fig: Matplotlib figure object with efficiency plot
    """
    # Calculate denominator (total MC tracks) with result caching
    cache_key_mc = f"mc_{x_type}_denominator"
    denominator = cache_results(
        compute_denominator,
        common_mc_info_dict,
        x_type, xbins,
        cache_path=cache_path,
        cache_key=cache_key_mc
    )
    print(f"mc denominator: {denominator}")
    
    # Calculate traditional method numerators and efficiencies (with caching)
    cache_key_trad_num = get_cache_key("trad", track_type, x_type, "numerator", is_charge=is_charge, is_charge_wrong=is_charge_wrong)
    finder_numerator_trad, fitter_numerator_trad = cache_results(
        compute_numerator,
        common_mc_info_dict, track_type, x_type, 
        track_dict_trad, track_mdc_dict_trad, track_kal_dict_trad, 
        xbins, is_charge=is_charge, is_charge_wrong=is_charge_wrong,
        cache_path=cache_path,
        cache_key=cache_key_trad_num
    )
    
    # Calculate traditional finder efficiency
    cache_key_trad_eff = get_cache_key("trad", track_type, x_type, "efficiency", is_charge=is_charge, is_charge_wrong=is_charge_wrong)
    eff_trad, err_trad, total_eff_trad, total_err_trad = cache_results(
        compute_efficiency,
        finder_numerator_trad, denominator,
        cache_path=cache_path,
        cache_key=cache_key_trad_eff
    )
    
    # Calculate traditional fitter efficiency
    cache_key_trad_fit_eff = get_cache_key("trad", track_type, x_type, "fit_efficiency", is_charge=is_charge, is_charge_wrong=is_charge_wrong)
    fit_eff_trad, fit_eff_err_trad, total_fit_eff_trad, total_fit_err_trad = cache_results(
        compute_efficiency,
        fitter_numerator_trad, denominator,
        cache_path=cache_path,
        cache_key=cache_key_trad_fit_eff
    )
    
    # Calculate ML method numerators and efficiencies (with caching)
    cache_key_ml_num = get_cache_key("ml", track_type, x_type, "numerator", is_charge=is_charge, is_charge_wrong=is_charge_wrong)
    finder_numerator_ml, fitter_numerator_ml = cache_results(
        compute_numerator,
        common_mc_info_dict, track_type, x_type, 
        track_dict_ml, track_mdc_dict_ml, track_kal_dict_ml,
        xbins, is_charge=is_charge, is_charge_wrong=is_charge_wrong,
        cache_path=cache_path,
        cache_key=cache_key_ml_num
    )
    
    # Calculate ML finder efficiency
    cache_key_ml_eff = get_cache_key("ml", track_type, x_type, "efficiency", is_charge=is_charge, is_charge_wrong=is_charge_wrong)
    eff_ml, err_ml, total_eff_ml, total_err_ml = cache_results(
        compute_efficiency,
        finder_numerator_ml, denominator,
        cache_path=cache_path,
        cache_key=cache_key_ml_eff
    )
    
    # Calculate ML fitter efficiency
    cache_key_ml_fit_eff = get_cache_key("ml", track_type, x_type, "fit_efficiency", is_charge=is_charge, is_charge_wrong=is_charge_wrong)
    fit_eff_ml, fit_eff_err_ml, total_fit_eff_ml, total_fit_err_ml = cache_results(
        compute_efficiency,
        fitter_numerator_ml, denominator,
        cache_path=cache_path,
        cache_key=cache_key_ml_fit_eff
    )
    
    # Create summary table for efficiency results
    table_data = [
        ["params", "trad", "ML", "unit"],
        ["Y", track_type, track_type, ""],
        ["X", x_type, x_type, ""],
        ["Finder eff", f"{total_eff_trad:.4f} ± {total_err_trad:.4f}", 
                  f"{total_eff_ml:.4f} ± {total_err_ml:.4f}", ""],
        ["Fitter eff", f"{total_fit_eff_trad:.4f} ± {total_fit_err_trad:.4f}", 
                    f"{total_fit_eff_ml:.4f} ± {total_fit_err_ml:.4f}", ""],
    ]

    bin_centers = (xbins[:-1] + xbins[1:]) / 2
    bin_widths = (xbins[1:] - xbins[:-1]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(5, 3)) 
    ax.set_position([0.18, 0.22, 0.75, 0.55])
    ax.set_box_aspect(0.6)
    ax.tick_params(axis='both', labelsize=8)
    ax.yaxis.set_label_coords(-0.22, 0.5, transform=ax.transAxes)
    
    # Plot efficiency curves
    plot_configs(ax, data_type, bin_centers, bin_widths, xlabel, ylim, 
                eff_trad, err_trad, eff_ml, err_ml,
                fit_eff_trad, fit_eff_err_trad,
                fit_eff_ml, fit_eff_err_ml,
                track_type=track_type,
                is_charge=is_charge,
                is_charge_wrong=is_charge_wrong,
                plot_finder=True,
                plot_fitter=True)
    

    ax.set_xlim(xbins[0], xbins[-1])

    return fig