import numpy as np

__all__ = [
    "z_remove",
    "to_pdf"
]

def z_remove(vals, max_z):
    s = np.std(vals)
    m = np.mean(vals)
    z = (vals - m)/s
    return vals[np.abs(z) < max_z]

def to_pdf(vals:np.ndarray, n_bins:int = 100, pseudocount:float = 0.0):
    
    hist, bin_edges = np.histogram(vals, bins=n_bins)
    
    if pseudocount > 0:
        pc = np.float64(np.zeros_like(hist)) + pseudocount
        hist = np.float64(hist) + pc
    
    total_counts = np.size(vals)
    
    return (hist/total_counts), bin_edges