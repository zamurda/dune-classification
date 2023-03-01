import numpy as np


__all__ = ["quality_cut", "rotate_pfo_2d"]


def _delete_particles(event_obj, idx_arr) -> None:
    #list of attributes
    attr_list = [a for a in dir(event_obj) if not a.startswith("__") and not callable(getattr(event_obj, a))]
    for attr in attr_list:
        setattr(event_obj, attr, np.delete((getattr(event_obj, attr)), idx_arr, 0))

def quality_cut(event_obj, event_num = False, var_names: tuple = ("purity", "completeness", "reco_num_hits_w"),
                reqs: tuple = (0.8, 0.8, 15)) -> None:
    #check if all variable names are valid and each variable has a valid requirement
    if all(var in dir(event_obj) for var in var_names) and len(var_names) == len(reqs):
        indices = np.array([])
        #first add indices where hit arrays are empty
        empty_mask = [np.any(i) for i in event_obj.reco_hits_x_w]
        indices = np.append(indices, [[i for i,val in enumerate(empty_mask) if not val]])
        
        if not event_num:
            #get indices where requirements are not met and pick out the unique indices
            for idx, attr in enumerate(var_names):
                indices = np.append(indices, np.where(getattr(event_obj, attr) < reqs[idx])[0])
            indices = np.unique(np.int64(indices))
            
            #del particles with these indices
            _delete_particles(event_obj, indices)
          
        else:
            #do same but only for records in a particular event_obj
            event_idxs = np.where(event_obj.event_number == event_num)
            for idx, attr in enumerate(var_names):
                np.concatenate(np.where(getattr(event_obj, attr)[event_idxs[0]:(event_idxs[-1]+1)] < reqs[idx])[0], indices)
            indices = np.unique(np.int64(indices))
            
            _delete_particles(event_obj, indices)
          
    else:
        raise RuntimeError("Variable names are invalid and/or each variable does not have a requirement")


def rotate_pfo_2d(event, n, v="w"):
    """Returns the hits in the vth view, rotated by an angle theta such
    that the transformed x-axis is strictly the direction of travel for the particle
    in that view.
    """
    if v.lower() == "w":
        vtx = np.array([
            event.reco_particle_vtx_x[n], event.reco_particle_vtx_w[n]
        ])
        X = np.concatenate((vtx.reshape(1,2),np.vstack((event.reco_hits_x_w[n], event.reco_hits_w[n])).T), axis=0) - vtx
        
    elif v.lower() == "u":
        vtx = np.array([
            event.reco_particle_vtx_x[n], event.reco_particle_vtx_u[n]
        ])
        X = np.concatenate((vtx.reshape(1,2),np.vstack((event.reco_hits_x_u[n], event.reco_hits_u[n])).T), axis=0) - vtx
        
    elif v.lower() == "v":
        vtx = np.array([
            event.reco_particle_vtx_x[n], event.reco_particle_vtx_v[n]
        ])
        X = np.concatenate((vtx.reshape(1,2),np.vstack((event.reco_hits_x_v[n], event.reco_hits_v[n])).T), axis=0) - vtx
    
    else:
        raise ValueError(
        "expected v to be in [u,v,w]"
        )
        
    xbar = np.mean(X, axis=0)
    xhat = xbar/np.linalg.norm(xbar)
    theta = -np.arctan2(xhat[1], xhat[0])

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    W = R @ X.T

    return W[0], W[1], xhat    