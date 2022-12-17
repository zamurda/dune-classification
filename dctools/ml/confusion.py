import numpy as np

__all__ = ["conf"]


def conf(pred, target, diagnostic=False):
    """
    Returns confusion matrix for a  binary classifier
    """
        
    rp = np.size(np.where(target==1)[0])
    rn = np.size(np.where(target==0)[0])
    tp = np.count_nonzero((pred==1) & (target==1))
    tn = np.count_nonzero((pred==0) & (target==0))
    
    tpr = tp/rp
    fnr = 1-tpr
    tnr = tn/rn
    fpr = 1-tnr
    
    precision = tp/(tp+(fpr*rn))
    
    return np.array([
        [tpr, fnr],
        [fpr, tnr]
    ]) if not diagnostic else (tpr,precision)