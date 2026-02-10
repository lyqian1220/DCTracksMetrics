from argparse import ArgumentParser

def get_parser():
    ap = ArgumentParser()

    
    ap.add_argument(
        '--data_type', 
        type=str, 
        default="Single-track",
        help='Category of event, will be used in plot text'
    )
    
    ap.add_argument(
        '--particle_type', 
        type=str, 
        choices=['e', 'muon', 'pi', 'K', 'p', 'dimu', 'flat', 'nby'],  # Restrict to valid particles
        default='pi',
        help='Type of particle'
    )
    
    ap.add_argument(
        '--particle_q', 
        type=str, 
        choices=['positive', 'negative'],  # Restrict to valid charges
        default='positive',
        help='Charge of particle, if multi-track, not set'
    )


    ap.add_argument(
        '--root_dir', 
        type=str, 
        required=True,
        help='Root directory for finder and fitter outputs'
    )
    
    ap.add_argument(
        '--trad_dir', 
        type=str, 
        required=True,
        help='Relative path of baseline fitting file to root_dir'
    )
    
    ap.add_argument(
        '--ml_dir', 
        type=str, 
        required=True,
        help='Relative path of ML fitting file to root_dir'
    )
    
    ap.add_argument(
        '--trad_mdc_dir', 
        type=str, 
        required=True,
        help='Relative path of baseline finding file to root_dir'
    )
    
    ap.add_argument(
        '--ml_mdc_dir', 
        type=str, 
        required=True,
        help='Relative path of ML finding file to root_dir'
    )
    
    ap.add_argument(
        '--isPhy', 
        type=bool, 
        default=False,
        help='Whether the events are physics events (default: False)'
    )

    ap.add_argument(
        '--cache_path', 
        type=str, 
        default="./cache",
        help='Root directory for cache files'
    )
    
    ap.add_argument(
        '--output_pdf', 
        type=str, 
        default="results_pdf/test.pdf",
        help='Path to output PDF file'
    )

    ap.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Number of files to process in each batch'
    )

    ap.add_argument(
        '--file_pattern', 
        type=str, 
        default="*.rec",
        help='File pattern to match (default: *.rec)')
    
    return ap