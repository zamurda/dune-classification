from uproot_io import Events
import numpy as np
import matplotlib.pyplot as plt
import inspect

def reco_display(event_obj, event_num, wire_plane:str):
    if isinstance(event_obj, Events):
        reco_planes = {
        "u" : [event_obj.reco_hits_x_u, event_obj.reco_hits_u, event_obj.reco_adcs_u],
        "v" : [event_obj.reco_hits_x_v, event_obj.reco_hits_v, event_obj.reco_adcs_v],
        "w" : [event_obj.reco_hits_x_w, event_obj.reco_hits_w, event_obj.reco_adcs_w]
        }
            
        if wire_plane in list(reco_planes):
            arr1 = event_obj.filter_by_event(reco_planes[wire_plane][0], event_num)
            arr2 = event_obj.filter_by_event(reco_planes[wire_plane][1], event_num)
            acds = event_obj.filter_by_event(reco_planes[wire_plane][2], event_num)
            
            for particle_idx in range(np.size(arr1)):
                plt.scatter(arr2[particle_idx], arr1[particle_idx], s=8,c=acds[particle_idx],cmap='plasma')
            
            plt.colorbar()
            plt.show()
            
        else:
            raise ValueError("Wire Plane is invalid")
        
    else:
        raise TypeError("Function expects uproot_io Events object type")

    
def norm_hist(vals, n, vline=None):
    if len(np.shape(vals)) > 1:
        for val in vals:
            plt.hist(val, density=True, bins=n, histtype="step")
    else:
        plt.hist(vals, density=True, bins=n, histtype="step")
    if vline is not None:
        plt.axvline(vline, ls="dotted", c="g")
    plt.show()
    
def particle_view(event_obj: Events, particle_id: int, x_axis: str, y_axis: str) -> None:
    xvals = getattr(event_obj, x_axis)[particle_id]
    yvals = getattr(event_obj, y_axis)[particle_id]
    #xmag = int(np.log10(xvals[0]))
    #ymag = int(np.log10(yvals[0]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    fig.suptitle(f"{y_axis} against {x_axis} for particle {particle_id}")
    
    ax1.set(xlabel= x_axis, ylabel= y_axis)
    ax1.set_label("Zoomed-in View")
    ax1.scatter(xvals, yvals)
    
    #ax2.set(xlabel= x_axis, ylabel= y_axis)
    #ax2.set_label("Normed Axis View")
    #ax2.set_xlim([0, 10**(xmag+1)])
    #ax2.set_ylim([0, 10**(ymag+1)])
    #ax2.scatter(xvals, yvals)
    
    fig.show()
    
    