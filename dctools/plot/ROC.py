import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


__all__ = [
    "ROC_divide",
    "ROC_curve",
    "plot_ROC"
]


def ROC_divide(tr,sh):
    minimum = min(min(tr),min(sh))
    maximum = max(max(tr),max(sh))
    divisions = 500
    ep_radius_max = 0

    for i in range(divisions+1):
        divide = minimum + i*(maximum - minimum)/divisions
        
        showers_as_shower = np.size([i for i in sh if i < divide])
        showers_as_track = np.size(sh) - showers_as_shower
        tracks_as_shower = np.size([i for i in tr if i < divide])
        tracks_as_track = np.size(tr) - tracks_as_shower
        
        efficiency = (tracks_as_track)/((tracks_as_track)+showers_as_track)
        purity = (tracks_as_track)/((tracks_as_track)+tracks_as_shower)

        ep_radius = efficiency**2 + purity**2
        
        if ep_radius > ep_radius_max:
            ep_radius_max = ep_radius
            best_divide = divide
            
    return best_divide

def ROC_curve(tr,sh):
    minimum = min(min(tr),min(sh))
    maximum = max(max(tr),max(sh))

    track_efficiency = np.array([])
    track_purity = np.array([])
    divisions = 2000
    ep_radius_max = 0

    for i in range(divisions+2):
        divide = minimum + i*(maximum - minimum)/divisions
        
        showers_as_shower = np.size([i for i in sh if i > divide])
        showers_as_track = np.size(sh) - showers_as_shower
        tracks_as_shower = np.size([i for i in tr if i > divide])
        tracks_as_track = np.size(tr) - tracks_as_shower
        
        efficiency = (tracks_as_track)/(tracks_as_track+showers_as_track)
        purity = (tracks_as_track)/(tracks_as_track+tracks_as_shower)
    
        ep_radius = efficiency+purity
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
            