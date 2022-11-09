import numpy as np
from uproot_io import Events
from numpy import poly1d

def path_ratio(event_obj: Events, deg: int):
    if isinstance(event_obj, Events):
        ratios = np.zeros_like(event_obj.reco_hits_w)
        for particle_idx, particle_hits in np.ndenumerate(event_obj.reco_particle_hits_w):
            fit_coeffs = np.polyfit(event_obj.reco_hits_x_w[particle_idx[0]], particle_hits, deg)[0]
            
            #apply func to yvals
            func = poly1d(fit_coeffs)
            yvals = np.zeros_like(event_obj.reco_hits_x_w[particle_idx])
            for j in len(yvals):
                yvals[j] = func(event_obj.reco_hits_x_w[particle_idx][j])
            
            #length of best fit
            f_prime = np.gradient(yvals)
            path_length = np.trapz(
                (np.sqrt(1 + np.gradient(event_obj.reco_hits_w, event_obj.reco_hits_x_w)) ** 2),
                event_obj.reco_hits_x_w
                )
            
            true_length = (particle_hits[-1] - particle_hits[0]) / (event_obj.reco_hits_x_w[particle_idx][-1] - event_obj.reco_hits_x_w[particle_idx][-1])
            
            ratios[particle_idx] = abs(path_length / true_length)
        
        return ratios
    else:
        raise TypeError(f"expected type uproot_io.Events, recieved {type(event_obj)}")