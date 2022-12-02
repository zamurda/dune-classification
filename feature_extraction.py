import numpy as np
from uproot_io import Events

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
    
    return [
        [tpr, fnr],
        [fpr, tnr]
    ] if not diagnostic else (tpr,precision)

def adc_res(event_obj: Events, iter: list):
    res_arr = np.array([])
    invalid = []
    for idx in iter:
        adc = event_obj.reco_adcs_w[idx]
        vtx = (event_obj.reco_particle_vtx_x[idx], event_obj.reco_particle_vtx_w[idx])
        xvals = event_obj.reco_hits_x_w[idx]
        yvals = event_obj.reco_hits_w[idx]        
        dist = np.array([])
        for point in zip(xvals, yvals):
            r = np.sqrt((point[0] - vtx[0])**2 + (point[1] - vtx[1])**2)
            dist = np.append(dist, r)
        
        
        if len(dist) != 0:
            fit = np.polyfit(dist, adc, 1, full=True)
            rms = np.sqrt(fit[1]/len(adc))
            res_arr = np.append(res_arr, rms)
        
        else:
            invalid.append(idx)
            continue
    
    return res_arr, invalid

def michel(event_obj: Events, iter: list):
    energies = np.array([])
    for idx in iter:
        adc = event_obj.reco_adcs_w[idx]
        tot = np.sum(adc)
        last = np.sum(adc[len(iter)-(len(iter)//10)-1:len(iter)])
        print(tot, last, last/tot)
        energies = np.append(energies, last/tot)
    return energies

class Features:
    def __init__(self, filename: str, create=False):
        pass