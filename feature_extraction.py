import numpy as np
from uproot_io import Events
from numpy import poly1d
from scipy.interpolate import interp1d

def conf(event_obj: Events, metric, cut, invalid_ids):
    #right of cut is shw
    actual = np.delete(event_obj.is_shower, invalid_ids)
    predicted = np.zeros_like(metric)
    predicted[np.where(metric > cut)] = 1
    
    ttat = 0
    ttas = 0
    tsat = 0
    tsas = 0
    for e in zip(actual, predicted):
        if e[0] == 0 and e[1] == 0:
            ttat += 1
        elif e[0] == 0 and e[1] == 1:
            ttas += 1
        elif e[0] == 1 and e[1] == 0:
            tsat += 1
        else:
            tsas += 1
    trk_correctness = ttat/len(np.where(actual == 0)[0])
    shw_correctness = tsas/len(np.where(actual == 1)[0])
    return [ttat, ttas, tsat, tsas], trk_correctness, shw_correctness

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