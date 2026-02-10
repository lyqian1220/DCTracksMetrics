import os
import sys
import gc
import tempfile
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import uproot
import ROOT

uproot.default_library = "np"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

# Import custom performance analysis modules
from src.performance import (
    track, hit, config, data_loader, resolution, parser
)


def merge_root_files_with_tchain(input_files, output_file, tree_name="events"):
    """
    Merge multiple ROOT files into a single file using ROOT.TChain
    """
    if not input_files:
        print("Error: No input files provided")
        return False
    
    print(f"Merging {len(input_files)} files into {output_file} using TChain")
    
    chain = ROOT.TChain(tree_name)
    
    # Add valid files to TChain
    for i, file_path in enumerate(input_files):
        if os.path.exists(file_path):
            chain.Add(file_path)
            print(f"  Added file {i+1}/{len(input_files)}: {Path(file_path).name}")
        else:
            print(f"  Warning: File does not exist {file_path}")
    
    total_entries = chain.GetEntries()
    print(f"  Total entries in chain: {total_entries:,}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create new file and clone tree
    outfile = ROOT.TFile(output_file, "RECREATE")
    new_tree = chain.CloneTree(0)
    new_tree.SetAutoFlush(10000)
    
    for i in range(total_entries):
        chain.GetEntry(i)
        new_tree.Fill()
        
        if i % 10000 == 0 and i > 0:
            progress = i / total_entries * 100
            print(f"    Processed {i:,}/{total_entries:,} events ({progress:.1f}%)")
    
    new_tree.Write()
    outfile.Close()
    
    print(f"  Merging completed! Output file: {output_file}")
    return True

def merge_data_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries with special handling for numpy arrays and lists
    """
    import numpy as np
    
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = value
        else:
            if isinstance(value, dict) and isinstance(dict1[key], dict):
                dict1[key] = merge_data_dicts(dict1[key], value)
            elif isinstance(value, list) and isinstance(dict1[key], list):
                dict1[key].extend(value)
            elif isinstance(value, np.ndarray) and isinstance(dict1[key], np.ndarray):
                dict1[key] = np.concatenate([dict1[key], value])
            else:
                dict1[key] = value
    
    return dict1

def process_file_batch(files_trad, files_ml, files_trad_mdc, files_ml_mdc, 
                       batch_idx, batch_cache_dir, tree_name="events", isPhy=False):
    """
    Process a single batch of files by first merging with TChain then loading data
    """
    print(f"Processing batch {batch_idx}...")
    
    # Create temporary directory for merged files
    temp_dir = tempfile.mkdtemp(prefix=f"batch_{batch_idx}_")
    
    try:
        # Merge baseline method files
        merged_trad = None
        if files_trad:
            merged_trad = os.path.join(temp_dir, "merged_trad.rec")
            merge_root_files_with_tchain(files_trad, merged_trad, tree_name)
        
        # Merge ML method files
        merged_ml = None
        if files_ml:
            merged_ml = os.path.join(temp_dir, "merged_ml.rec")
            merge_root_files_with_tchain(files_ml, merged_ml, tree_name)
        
        # Merge baseline MDC files
        merged_trad_mdc = None
        if files_trad_mdc:
            merged_trad_mdc = os.path.join(temp_dir, "merged_trad_mdc.rec")
            merge_root_files_with_tchain(files_trad_mdc, merged_trad_mdc, tree_name)
        
        # Merge ML MDC files
        merged_ml_mdc = None
        if files_ml_mdc:
            merged_ml_mdc = os.path.join(temp_dir, "merged_ml_mdc.rec")
            merge_root_files_with_tchain(files_ml_mdc, merged_ml_mdc, tree_name)
        
        data_MDC = None
        if merged_trad_mdc and merged_ml_mdc:
            data_MDC = data_loader.load_all_data(
                merged_trad_mdc,
                merged_ml_mdc,
                isMDC=True,
                isPhy=isPhy,
                cache_path=os.path.join(batch_cache_dir, f"mdc_batch_{batch_idx}"),
            )

        data = None
        if merged_trad and merged_ml:
            data = data_loader.load_all_data(
                merged_trad,
                merged_ml,
                isMDC=False,
                isPhy=isPhy,
                cache_path=os.path.join(batch_cache_dir, f"batch_{batch_idx}"),
            )
        
        return data, data_MDC
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            gc.collect()

def process_file_batch_direct(files_trad, files_ml, files_trad_mdc, files_ml_mdc,
                              batch_idx, batch_cache_dir, isPhy=False):
    """
    Process a batch of files directly (without merging) to save disk space
    Loads each file individually then merges data dictionaries in memory
    """
    print(f"Processing batch {batch_idx} (direct method)...")
    
    batch_data_list = []
    batch_data_mdc_list = []
    
    # Process each file pair in the batch
    for i, (file_trad, file_ml) in enumerate(zip(files_trad, files_ml)):
        file_idx = batch_idx * len(files_trad) + i + 1
        print(f"  Processing file {file_idx}: {Path(file_trad).name}, {Path(file_ml).name}")
        
        # Load main data for current file pair
        data = data_loader.load_all_data(
            file_trad,
            file_ml,
            isMDC=False,
            isPhy=isPhy,
            cache_path=os.path.join(batch_cache_dir, f"file_{file_idx}"),
        )
        batch_data_list.append(data)
        
        if i < len(files_trad_mdc) and i < len(files_ml_mdc):
            data_MDC = data_loader.load_all_data(
                files_trad_mdc[i],
                files_ml_mdc[i],
                isMDC=True,
                isPhy=isPhy,
                cache_path=os.path.join(batch_cache_dir, f"file_{file_idx}_mdc"),
            )
            batch_data_mdc_list.append(data_MDC)
        
        if i % 2 == 0:
            gc.collect()
    
    merged_batch_data = None
    for data in batch_data_list:
        if merged_batch_data is None:
            merged_batch_data = data
        else:
            merged_batch_data = merge_data_dicts(merged_batch_data, data)
    
    # Merge MDC data from all files in batch
    merged_batch_data_mdc = None
    for data_mdc in batch_data_mdc_list:
        if merged_batch_data_mdc is None:
            merged_batch_data_mdc = data_mdc
        else:
            merged_batch_data_mdc = merge_data_dicts(merged_batch_data_mdc, data_mdc)
    
    return merged_batch_data, merged_batch_data_mdc

def generate_plots_batch(
    data_type,
    files_trad_list,
    files_ml_list,
    files_trad_mdc_list,
    files_ml_mdc_list,
    cache_path,
    output_pdf,
    batch_size=2,
    use_direct_method=True,
    isPhy=False
):
    """
    Process multiple files in batches and generate comprehensive analysis PDF
    """
    # Create batch cache directory
    batch_cache_dir = os.path.join("./output/cache", cache_path)
    os.makedirs(batch_cache_dir, exist_ok=True)
    
    # Calculate number of batches needed
    total_files = len(files_trad_list)
    num_batches = (total_files + batch_size - 1) // batch_size
    
    print(f"Total files: {total_files}, batches: {num_batches}, files per batch: {batch_size}")
    print(f"Processing method: {'direct processing' if use_direct_method else 'merge then process'}")
    
    all_data_batches = []
    all_data_mdc_batches = []
    
    # Process each batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx+1}/{num_batches}: files {start_idx+1}-{end_idx}")
        print(f"{'='*60}")
        
        # Get files for current batch
        batch_files_trad = files_trad_list[start_idx:end_idx]
        batch_files_ml = files_ml_list[start_idx:end_idx]
        batch_files_trad_mdc = files_trad_mdc_list[start_idx:end_idx]
        batch_files_ml_mdc = files_ml_mdc_list[start_idx:end_idx]
        
        # Process batch using selected method
        if use_direct_method:
            data, data_MDC = process_file_batch_direct(
                batch_files_trad, batch_files_ml,
                batch_files_trad_mdc, batch_files_ml_mdc,
                batch_idx, batch_cache_dir, isPhy
            )
        else:
            data, data_MDC = process_file_batch(
                batch_files_trad, batch_files_ml,
                batch_files_trad_mdc, batch_files_ml_mdc,
                batch_idx, batch_cache_dir, isPhy
            )
        
        # Save batch data for merging
        if data:
            all_data_batches.append(data)
        if data_MDC:
            all_data_mdc_batches.append(data_MDC)
        
        # Release memory after each batch
        gc.collect()
    
    # Merge all batch data into final dataset
    print(f"\nMerging all batch data...")
    
    # Create final cache directory
    final_cache_path = os.path.join("./output/cache", cache_path, "merged_final")
    os.makedirs(final_cache_path, exist_ok=True)
    
    # Merge main data
    merged_data = None
    if all_data_batches:
        merged_data = all_data_batches[0]
        for i in range(1, len(all_data_batches)):
            merged_data = merge_data_dicts(merged_data, all_data_batches[i])
            if i % 2 == 0:
                gc.collect()
    
    # Merge MDC data
    merged_data_MDC = None
    if all_data_mdc_batches:
        merged_data_MDC = all_data_mdc_batches[0]
        for i in range(1, len(all_data_mdc_batches)):
            merged_data_MDC = merge_data_dicts(merged_data_MDC, all_data_mdc_batches[i])
            if i % 2 == 0:
                gc.collect()
    
    # Generate plots from merged data
    generate_plots_from_merged_data(
        data_type,
        merged_data,
        merged_data_MDC,
        output_pdf,
        final_cache_path
    )
    
    print(f"✓ Processing complete! PDF saved to: {output_pdf}")

def generate_plots_from_merged_data(
    data_type,
    merged_data,
    merged_data_MDC,
    output_pdf,
    cache_path
):
    """
    Generate comprehensive analysis plots from merged data and save to PDF
    
    Generates:
    1. Hit purity/efficiency distribution plots
    2. Resolution plots (MDC and Kal tracks)
    3. Track efficiency plots for different track types
    4. Hit efficiency/purity violin plots
    
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_pdf)
    os.makedirs(output_dir, exist_ok=True)
    
    if merged_data is None:
        print("Error: No merged data available for plotting")
        return
    
    # Track type configurations for efficiency plots
    track_types_1 = ["Mdc", "Kal"]
    track_types_2 = ["Mdc"]
    hit_data_types = {
        "eff_pt", "eff_cos", "eff_phi", "purity_pt", "purity_cos", "purity_phi"
    }
    # Physics parameters to plot resolution for
    param_names = {"dr", "phi0", "kappa", "dz", "tanl", "px", "py", "pz", "pT", "p"}
    
    # Track type mapping for efficiency plot configuration
    track_type_map = {
        "Good": (False, False, merged_data["good_track_dict"]["trad"], merged_data["good_track_dict"]["ml"], (0., 1.2)),
        "Charge": (True, False, merged_data["good_track_dict"]["trad"], merged_data["good_track_dict"]["ml"], (0., 1.2)),
        "ChargeWrong": (False, True, merged_data["good_track_dict"]["trad"], merged_data["good_track_dict"]["ml"], (-0.04, 0.04)),
        "Fake": (False, False, merged_data["fake_track_dict"]["trad"], merged_data["fake_track_dict"]["ml"], (-0.04, 0.04)) if "fake_track_dict" in merged_data else (False, False, {}, {}, (-0.04, 0.04)),
        "Clone": (False, False, merged_data["clone_track_dict"]["trad"], merged_data["clone_track_dict"]["ml"], (-0.02, 0.02)) if "clone_track_dict" in merged_data else (False, False, {}, {}, (-0.02, 0.02))
    }
   
    # Create PDF file and generate plots
    with PdfPages(output_pdf) as pdf:
        # Process 1: Resolution plots (MDC and Kal tracks)
        print("Process1: Generating resolution plots (MDC and Kal tracks)...")
        total_plots = len(param_names) * (len(track_types_1) + len(track_types_2))
        current_plot = 1

        print("Generating dual MDC-Kal resolution plots...")
        for param_idx, param_name in enumerate(param_names, 1):
            print(f"  Dual plot {param_idx}/{len(param_names)} : {param_name}")

            # Generate resolution plots for current parameter
            fig1, fig2 = resolution.plot_resolution_dual_mdc_kal(
                data_type=data_type,
                common_mc_dict     = merged_data["mc_dict_filtered"]["common"],
                track_dict_kal_trad   = merged_data["track_info_dict"].get("Kal", {}).get("trad", {}),
                track_dict_kal_ml     = merged_data["track_info_dict"].get("Kal", {}).get("ml", {}),
                good_track_dict_kal_trad = merged_data["good_track_dict"]["trad"],
                good_track_dict_kal_ml   = merged_data["good_track_dict"]["ml"],
                track_dict_mdc_trad   = merged_data_MDC["track_info_dict"].get("Mdc", {}).get("trad", {}) if merged_data_MDC else {},
                track_dict_mdc_ml     = merged_data_MDC["track_info_dict"].get("Mdc", {}).get("ml", {}) if merged_data_MDC else {},
                good_track_dict_mdc_trad = merged_data_MDC["good_track_dict"]["trad"] if merged_data_MDC else {},
                good_track_dict_mdc_ml   = merged_data_MDC["good_track_dict"]["ml"] if merged_data_MDC else {},
                param_name=param_name,
                cache_path=cache_path
            )

            for fig in [fig1, fig2]:
                pdf.savefig(fig, bbox_inches='tight')

            plt.close(fig1)
            plt.close(fig2)

    
        # Process 2: Track efficiency plots
        print("\nProcess2: Generating track efficiency plots...")
        total_efficiency_plots = len(config.xvar_configs) * len(track_type_map)
        current_eff_plot = 1
        
        for x_idx, (x_type, x_params) in enumerate(config.xvar_configs.items(), 1):
            print(f"\n  Processing x-variable group {x_idx}/{len(config.xvar_configs)}: {x_type}")
            
            # Iterate over all track types
            for track_idx, track_type in enumerate(track_type_map.keys(), 1):
                progress = f"Plot {current_eff_plot}/{total_efficiency_plots}"
                print(f"    {progress}: x-type={x_type}, Track type={track_type}")

                is_charge, is_charge_wrong, track_dict_trad, track_dict_ml, ylim = track_type_map[track_type]
                
                # Skip if track data is empty
                if not track_dict_trad or not track_dict_ml:
                    print(f"    Skipping: {track_type} track_dict is empty")
                    current_eff_plot += 1
                    continue

                # Generate efficiency plot
                fig = track.plot_efficiency(
                    common_mc_info_dict=merged_data["mc_dict_filtered"]["common"],
                    data_type=data_type,
                    track_type=track_type,
                    is_charge=is_charge,
                    is_charge_wrong=is_charge_wrong,
                    track_dict_trad=track_dict_trad,
                    track_dict_ml=track_dict_ml,
                    track_mdc_dict_trad=merged_data["track_info_dict"]["Mdc"]["trad"],
                    track_mdc_dict_ml=merged_data["track_info_dict"]["Mdc"]["ml"],
                    track_kal_dict_trad=merged_data["track_info_dict"]["Kal"]["trad"] if "Kal" in merged_data["track_info_dict"] else {},
                    track_kal_dict_ml=merged_data["track_info_dict"]["Kal"]["ml"] if "Kal" in merged_data["track_info_dict"] else {},
                    x_type=x_type,
                    xbins=x_params["bins"],
                    xlabel=x_params["xlabel"],
                    ylim=ylim,
                    cache_path=cache_path
                )
                pdf.savefig(fig)
                plt.close(fig)
                current_eff_plot += 1
        
        # Process 3: Hit efficiency and purity violin plots
        if "hit_violin_data" in merged_data:
            print("\nProcess3: Generating hit efficiency and purity violin plots...")
            total_violin_plots = len(hit_data_types)
            current_violin_plot = 1
            
            for y_type in hit_data_types:
                if y_type in merged_data["hit_violin_data"]:
                    print(f"  Plot {current_violin_plot}/{total_violin_plots}: Yaxis type={y_type}")

                    data_config = config.hit_violin_config(y_type)
                    raw_data = merged_data["hit_violin_data"][y_type]

                    merged_by_label = {"Baseline Finder": None, "GNN Finder": None}
                    for label, values, bin_values, color in raw_data:
                        if label not in merged_by_label:
                            continue  
                        if merged_by_label[label] is None:
                            # Initialize with copy of data
                            merged_by_label[label] = (label, values[:], bin_values[:], color)
                        else:
                            # Merge existing data with new values
                            old_label, old_values, old_bin_values, old_color = merged_by_label[label]
                            new_values = old_values + values
                            new_bin_values = old_bin_values + bin_values
                            merged_by_label[label] = (label, new_values, new_bin_values, old_color)

                    unique_data = [merged_by_label["Baseline Finder"], merged_by_label["GNN Finder"]]
                    unique_data = [item for item in unique_data if item is not None]

                    total_original = sum(len(item[1]) for item in raw_data)
                    total_merged = sum(len(item[1]) for item in unique_data)
                    print(f"{y_type}: Original {len(raw_data)} entries ({total_original} tracks) → "
                          f"Merged {len(unique_data)} entries ({total_merged} tracks)")

                    # Generate violin plot
                    fig = hit.plot_violin(
                        data=unique_data,
                        bins=data_config["bins"],
                        widths=data_config["widths"],
                        xlabel=data_config["xlabel"],
                        ylabel=data_config["ylabel"],
                        ylim=data_config["ylim"],
                        yticks=data_config["yticks"],
                        data_type=data_type,
                    )
                    pdf.savefig(fig)
                    
                    eps_save_dir_hit = os.path.join(cache_path, "hit")
                    os.makedirs(eps_save_dir_hit, exist_ok=True) 
                    eps_filename = f"violin_plot_y{y_type}_idx{current_violin_plot:03d}.eps"
                    eps_file = os.path.join(eps_save_dir_hit, eps_filename)
                    fig.savefig(eps_file, format='eps', bbox_inches='tight')
                    
                    plt.close(fig)
                else:
                    print(f"  Skipping: {y_type} not found in hit_violin_data")
                
                current_violin_plot += 1

def main():
    
    ap = parser.get_parser()
    args = ap.parse_args()
    
    data_type = config.get_data_type(
        data_type=args.data_type,
        particle_type=args.particle_type,
        charge=args.particle_q
    )
    
    if (hasattr(args, 'trad_dir') and args.trad_dir and
        hasattr(args, 'ml_dir') and args.ml_dir):
        
        print("=" * 60)
        print("Automatically scanning directories for input files...")
        print("=" * 60)
        
        def scan_directory(base_dir, sub_dir, pattern="*.rec"):
            """
            Scan directory for files matching pattern
            """
            if not sub_dir:
                return []
            
            if sub_dir.startswith('/'):
                full_dir = sub_dir
            else:
                full_dir = os.path.join(base_dir, sub_dir) if base_dir else sub_dir
            
            if not os.path.isdir(full_dir):
                print(f"Warning: Directory does not exist {full_dir}")
                return []
            
            import glob
            files = sorted(glob.glob(os.path.join(full_dir, pattern)))
            print(f"  Scanned {full_dir}: found {len(files)} files")
            return files
        
        # Scan directories for input files
        files_trad_list = scan_directory(args.root_dir, args.trad_dir, args.file_pattern)
        files_ml_list = scan_directory(args.root_dir, args.ml_dir, args.file_pattern)
        files_trad_mdc_list = scan_directory(args.root_dir, args.trad_mdc_dir, args.file_pattern) if args.trad_mdc_dir else []
        files_ml_mdc_list = scan_directory(args.root_dir, args.ml_mdc_dir, args.file_pattern) if args.ml_mdc_dir else []
        
        if len(files_trad_list) == 0 or len(files_ml_list) == 0:
            print("Error: No files found in input directories!")
            return

        # Process files in batches and generate plots
        generate_plots_batch(
            data_type=data_type,
            files_trad_list=files_trad_list,
            files_ml_list=files_ml_list,
            files_trad_mdc_list=files_trad_mdc_list,
            files_ml_mdc_list=files_ml_mdc_list,
            cache_path=args.cache_path,
            output_pdf=args.output_pdf,
            batch_size=args.batch_size,
            use_direct_method=True,
            isPhy=args.isPhy
        )
        
    else:
        file_trad = f"{args.root_dir}/{args.file_trad}"
        file_ml = f"{args.root_dir}/{args.file_ml}"
        file_trad_MDC = f"{args.root_dir}/{args.file_trad_mdc}"
        file_ml_MDC = f"{args.root_dir}/{args.file_ml_mdc}"
        
        generate_plots(
            data_type=data_type,
            file_trad=file_trad,
            file_ml=file_ml,
            file_trad_MDC=file_trad_MDC,
            file_ml_MDC=file_ml_MDC,
            cache_path=args.cache_path,
            output_pdf=args.output_pdf,
            isPhy=args.isPhy
        )

if __name__ == "__main__":
    main()