import pybes3 as p3
import joblib
import os
import uproot

uproot.default_library = "np"

def load_evt(file):
    f = p3.open(file)
    evt = f["Event"]
    return evt

def get_cache_key(evt_type="trad", track_type="Good", x_type="pT", data_type="numerator", is_charge=False, is_charge_wrong=False):
    
    return f"{evt_type}_{track_type}_track_{x_type}_{data_type}_{is_charge}_{is_charge_wrong}"

def cache_results(func, *args, cache_path, cache_key, **kwargs):
    
    os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}...")
        return joblib.load(cache_file)
    
    print(f"!!!!!!Cache missed for {cache_key}, computing results...")
    results = func(*args, **kwargs)
    
    joblib.dump(results, cache_file)
    print(f"Saved results to cache: {cache_file} ----------------!")
    return results