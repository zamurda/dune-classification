from uproot_io import Events, View
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def ROC_divide(tr,sh):
    minimum = min(min(tr),min(sh))
    maximum = max(max(tr),max(sh))
    track = np.sort(tr)
    shower = np.sort(sh)
    divisions = 500
    ep_radius_max = 0

    for i in range(divisions+1):
        divide = minimum + i*(maximum - minimum)/divisions
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

def ROC_curve(tr,sh):
    minimum = min(min(tr),min(sh))
    maximum = max(max(tr),max(sh))
    track = np.sort(tr)
    shower = np.sort(sh)

    track_efficiency = np.array([])
    track_purity = np.array([])
    divisions = 400
    ep_radius_max = 0

    for i in range(divisions+1):
        divide = minimum + i*(maximum - minimum)/divisions
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
            best_efficiency = efficiency
            best_purity = purity

        track_efficiency = np.append(track_efficiency,efficiency)
        track_purity = np.append(track_purity,purity)

    
            
    return track_efficiency, track_purity, best_divide, best_efficiency, best_purity

def plot_ROC(tr,sh):
    eff, pur, divide, be, bp = ROC_curve(tr,sh)
    print("Best divide is at",divide)
    print("Best efficiency is",be,"Best purity is",bp)
    plt.xlabel("Track Purity")
    plt.ylabel("Track Efficiency")
    plt.scatter(pur,eff,s = 8)
            