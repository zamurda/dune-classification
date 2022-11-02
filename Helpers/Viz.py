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
            '''
            arr1 = np.hstack(event_obj.filter_by_event(reco_planes[wire_plane][0], event_num))
            arr2 = np.hstack(event_obj.filter_by_event(reco_planes[wire_plane][1], event_num))
            acds = np.hstack(event_obj.filter_by_event(reco_planes[wire_plane][2], event_num))
            

            plt.scatter(arr1, arr2,s=8,c=acds,cmap='hot')
            plt.colorbar()
            plt.show()
            '''
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


    def mc_display():
        pass