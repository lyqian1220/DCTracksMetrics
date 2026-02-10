import numpy as np

xvar_configs = {
    "pT": {
        "bins": np.linspace(0.15, 1.5, 11),
        "xlabel": r"$p_T^{\text{MC}}$ [GeV/$c$]",
    },
    "cos": {
        "bins": np.linspace(-0.93, 0.93, 11),
        "xlabel": r"$\cos\theta^{\text{MC}}$",
    },
    "phi": {
        "bins": np.linspace(-np.pi, np.pi, 13),
        "xlabel": r"$\phi^{\text{MC}}$ [rad]",
    },
}
def resolution_config(param_name):
    configs = {
        "dr": {
            "range_min": -0.1, 
            "range_max": 0.1, 
            "num_bins": 50,
            "c_num_bins": 100,
            "diff_num_bins": 200,
            "diff_range_min": -10, 
            "diff_range_max": 10, 
            "resolution_ylim": [1e-10, 1e-1],
            "xlabel": "[cm]"
        },
        "phi0": {
            "range_min": -1.0, 
            "range_max": 7.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.01, 
            "diff_range_max": 0.01, 
            "diff_num_bins": 100,
            "xlabel": "[rad]"
        },
        "kappa": {
            "range_min": -10.0, 
            "range_max": 10.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.1, 
            "diff_range_max": 0.1, 
            "diff_num_bins": 200,
            "resolution_ylim": [1e-3, 1.0],
            "xlabel": "[GeV⁻¹]"
        },
        "dz": {
            "range_min": -4.0, 
            "range_max": 4.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -3, 
            "diff_range_max": 3, 
            "diff_num_bins": 200,
            "resolution_ylim": [1e-10, 1e-1],
            "xlabel": "[cm]"
        },
        "tanl": {
            "range_min": -4.0, 
            "range_max": 4.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.2, 
            "diff_range_max": 0.2, 
            "diff_num_bins": 300,
            "xlabel": ""
        },
        "px": {
            "range_min": -2.0, 
            "range_max": 2.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.15, 
            "diff_range_max": 0.15, 
            "diff_num_bins": 200,
            "xlabel": r"[GeV/$c$]"
        },
        "py": {
            "range_min": -2.0, 
            "range_max": 2.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.2, 
            "diff_range_max": 0.2, 
            "diff_num_bins": 300,
            "xlabel": r"[GeV/$c$]"
        },
        "pz": {
            "range_min": -4.0, 
            "range_max": 4.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.15, 
            "diff_range_max": 0.15, 
            "diff_num_bins": 300,
            "xlabel": r"[GeV/$c$]"
        },
        "pT": {
            "range_min": 0.15, 
            "range_max": 1.5, 
            "num_bins": 100,
            "c_num_bins": 15,
            "diff_range_min": -0.06, 
            "diff_range_max": 0.06, 
            "diff_num_bins": 100,
            "resolution_ylim": [1e-3, 1.0],
            "xlabel": r"[GeV/$c$]"
        },
        "p": {
            "range_min": 0.0, 
            "range_max": 5.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.05, 
            "diff_range_max": 0.05, 
            "diff_num_bins": 200,
            "xlabel": r"[GeV/$c$]"
        }
    }

    default_config = {
        "range_min": -1.0, 
        "range_max": 1.0, 
        "num_bins": 100,
        "diff_range_min": -0.1, 
        "diff_range_max": 0.1, 
        "diff_num_bins": 100,
        "xlabel": ""
    }

    return configs.get(param_name, default_config)

def resolution_config_MDC(param_name):
    configs = {
        "dr": {
            "range_min": -0.2, 
            "range_max": 0.2, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -5, 
            "diff_range_max": 5, 
            "diff_num_bins": 100,
            "xlabel": "[cm]"
        },
        "phi0": {
            "range_min": -1.0, 
            "range_max": 7.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.02, 
            "diff_range_max": 0.02, 
            "diff_num_bins": 100,
            "xlabel": "[rad]"
        },
        "kappa": {
            "range_min": -10.0, 
            "range_max": 10.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.2, 
            "diff_range_max": 0.2, 
            "diff_num_bins": 100,
            "xlabel": "[GeV⁻¹]"
        },
        "dz": {
            "range_min": -4.0, 
            "range_max": 4.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -5, 
            "diff_range_max": 5, 
            "diff_num_bins": 100,
            "xlabel": "[cm]"
        },
        "tanl": {
            "range_min": -4.0, 
            "range_max": 4.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.5, 
            "diff_range_max": 0.5, 
            "diff_num_bins": 100,
            "xlabel": ""
        },
        "px": {
            "range_min": -2.0, 
            "range_max": 2.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.5, 
            "diff_range_max": 0.5, 
            "diff_num_bins": 100,
            "xlabel": "[GeV/c]"
        },
        "py": {
            "range_min": -2.0, 
            "range_max": 2.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.5, 
            "diff_range_max": 0.5, 
            "diff_num_bins": 100,
            "xlabel": "[GeV/c]"
        },
        "pz": {
            "range_min": -4.0, 
            "range_max": 4.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.5, 
            "diff_range_max": 0.5, 
            "diff_num_bins": 100,
            "xlabel": "[GeV/c]"
        },
        "pT": {
            "range_min": 0.15, 
            "range_max": 1.5, 
            "num_bins": 100,
            "c_num_bins": 15,
            "diff_range_min": -0.2, 
            "diff_range_max": 0.2, 
            "diff_num_bins": 100,
            "xlabel": "[GeV/c]"
        },
        "p": {
            "range_min": 0.0, 
            "range_max": 5.0, 
            "num_bins": 100,
            "c_num_bins": 100,
            "diff_range_min": -0.2, 
            "diff_range_max": 0.2, 
            "diff_num_bins": 100,
            "xlabel": "[GeV/c]"
        }
    }

    default_config = {
        "range_min": -1.0, 
        "range_max": 1.0, 
        "num_bins": 100,
        "diff_range_min": -0.1, 
        "diff_range_max": 0.1, 
        "diff_num_bins": 100,
        "xlabel": ""
    }

    return configs.get(param_name, default_config)

def hit_violin_config(data_type):
    widths = 0.1  
    ylim = (0.3, 1.1)
    yticks = np.arange(0.3, 1.1, 0.1)
    pt_bins = np.linspace(start=0.15, stop=1.5, num=11, endpoint=True)
    cos_bins = np.array([-0.93, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.93])
    phi_bins = np.linspace(-np.pi, np.pi, 11)
    
    configs = {
        "eff_pt": {
            "bins": pt_bins,  
            "widths": widths,
            "xlabel": r"$p_T^{\text{MC}}$ [GeV/$c$]",
            "ylabel": "Hit Efficiency",
            "ylim": ylim,
            "yticks": yticks,
        },
        "purity_pt": {
            "bins": pt_bins,
            "widths": widths,
            "xlabel": r"$p_T^{\text{MC}}$ [GeV/$c$]",
            "ylabel": "Hit Purity",
            "ylim": (0.85, 1.02),
            "yticks": np.arange(0.85, 1.02, 0.02),
        },
        "eff_cos": {
            "bins": cos_bins,  
            "widths": widths,
            "xlabel": r"$\cos\theta^{\text{MC}}$",
            "ylabel": "Hit Efficiency",
            "ylim": ylim,
            "yticks": yticks,
        },
        "purity_cos": {
            "bins": cos_bins,
            "widths": widths,
            "xlabel": r"$\cos\theta^{\text{MC}}$",
            "ylabel": "Hit Purity",
            "ylim": (0.85, 1.02),
            "yticks": np.arange(0.85, 1.02, 0.02),
        },
        "eff_phi": {
            "bins": phi_bins,
            "widths": 0.4,
            "xlabel": r"$\phi^{\text{MC}}$",
            "ylabel": "Hit Efficiency",
            "ylim": (0.6, 1.04),
            "yticks": np.arange(0.6, 1.04, 0.02),
        },
        "purity_phi": {
            "bins": phi_bins,
            "widths": 0.4,
            "xlabel": r"$\phi^{\text{MC}}$",
            "ylabel": "Hit Purity",
            "ylim": (0.8, 1.02),
            "yticks": np.arange(0.8, 1.02, 0.02),
        }
    }

    return configs.get(data_type, {})


particle_types = {
    'e': r'$e$',          
    'muon': r'$\mu$',     
    'pi': r'$\pi$',       
    'K': r'$K$',          
    'p': r'$p$',   
    'dimu': r'$\mu^+\mu^-$', 
    'flat': r'$\pi^+\pi^-$',
    'nby': r'$\pi^+\pi^-$'
}

charge_types = {
    'positive': {
        'default': r'$^+$',    
        'p': r''             
    },
    'negative': {
        'default': r'$^-$',  
        'p': r'$\overline{p}$' 
    }
}

def get_data_type(data_type, particle_type, charge):
    """
    Generate formatted data_type string with configurable category, particle symbol and charge
    
    Parameters:
        particle_category: Category of particle, e.g., "SingleParticle", "MultiParticle"...
        particle: Particle type, choose from: 'e', 'muon', 'pi', 'K', 'p'
        charge: Charge type, choose from: 'positive', 'negative'
    

    """
    # Validate particle input
    if particle_type not in particle_types:
        raise ValueError(f"Particle type {particle} not supported. Choose from: {list(particle_types.keys())}")
    
    # Validate charge input
    if charge not in charge_types:
        raise ValueError(f"Charge type {charge} not supported. Choose from: {list(charge_types.keys())}")

    # Get appropriate charge symbol (special case for proton)
    if particle_type == 'p':
        charge_type = charge_types[charge]['p']
    else:
        charge_type = charge_types[charge]['default']
    
    # Combine components: category + particle symbol + charge symbol
    if particle_type == 'p' and charge == 'negative':
        # For antiproton, use the overline symbol directly
        return f"{data_type} {charge_types[charge][particle_type]}"
    elif particle_type == 'dimu' and charge == 'positive':
        return f"{data_type} {particle_types[particle_type]}"
    elif particle_type == 'flat' and charge == 'positive':
        return f"{data_type} {particle_types[particle_type]}"
    elif particle_type == 'nby' and charge == 'positive':
        return f"{data_type} {particle_types[particle_type]}"
    else:
        # For other particles, combine base symbol and charge
        return f"{data_type} {particle_types[particle_type]}{charge_type}"