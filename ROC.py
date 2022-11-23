from uproot_io import Events, View
import numpy as np

def ROC_divide(tr,sh):
    max_track = max(tr)
    min_shower = min(sh)
    track = np.sort(tr)
    shower = np.sort(sh)
    divisions = 500
    ep_radius_max = 0

    for i in range(divisions+1):
        divide = min_shower + i*((max_track-min_shower)/divisions)
        for j in range(np.size(track)):
            if track[j] > divide:
                track_astrack = j
                track_asshower = np.size(track) - j
                break

        for j in range(np.size(shower)):
            if shower[j] > divide:
                shower_astrack = j
                shower_asshower = np.size(shower)-j
                break
        
        efficiency = (track_astrack)/(track_astrack+shower_astrack)
        purity = (track_astrack)/(track_astrack+track_asshower)

    
        ep_radius = efficiency**2 + purity**2
        
        if ep_radius > ep_radius_max:
            ep_radius_max = ep_radius
            best_divide = divide
        
    return best_divide
    
            