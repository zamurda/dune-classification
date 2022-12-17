import numpy as np

__all__ = [
    "idx_in_event",
    "pdg_in_event",
    "pdg_idx",
    "tracks_idx",
    "showers_idx"
]

particle_id_dict = {-2212:"ANTIPROTON", -321:"KAON -", -211:"PION -", -13:"MUON -", -11:"ELECTRON", 0:"NO BEST MATCH", 11:"POSTIRON", 13:"MUON +", 22:"GAMMA", 211:"PION +", 321:"KAON +", 2212:"PROTON", 3112:"SIGMA -", 3222:"SIGMA +"}

def idx_in_event(event_obj,number_event,direction,min_hits,purity):
    temp = np.where(event_obj.event_number == number_event)[0]
    if direction.lower() == "w":
        idx = [i for i in temp if event_obj.reco_num_hits_w[i] > min_hits and event_obj.purity[i] >= purity]
    if direction.lower() == "v":
        idx = [i for i in temp if event_obj.reco_num_hits_v[i] > min_hits and event_obj.purity[i] >= purity]
    if direction.lower() == "u":
        idx = [i for i in temp if event_obj.reco_num_hits_u[i] > min_hits and event_obj.purity[i] >= purity]
    return idx

def pdg_in_event(event_obj,number_event,direction,min_hits,purity):
    particle_pdg = [event_obj.mc_pdg[i] for i in idx_in_event(event_obj,number_event,direction,min_hits,purity)]
    particle_type = [particle_id_dict[event_obj.mc_pdg[i]] for i in idx_in_event(event_obj,number_event,direction,min_hits,purity)]
    return particle_pdg,particle_type

def pdg_idx(event_obj,pdg_code,min_hits,purity):
    temp = np.where(event_obj.mc_pdg == pdg_code)[0]
    idx = [i for i in temp if (event_obj.reco_num_hits_w[i] >= min_hits) and (event_obj.reco_num_hits_u[i] >= min_hits) and (event_obj.reco_num_hits_v[i] >= min_hits) and (event_obj.purity[i] >= purity)]
    return idx

def tracks_idx(event_obj,min_hits,purity):
    temp = np.where(event_obj.is_track == 1)[0]
    idx = [i for i in temp if (event_obj.reco_num_hits_w[i] >= min_hits) and (event_obj.reco_num_hits_u[i] >= min_hits) and (event_obj.reco_num_hits_v[i] >= min_hits) and (event_obj.purity[i] >= purity)]    
    return idx

def showers_idx(event_obj,min_hits,purity):
    temp = np.where(event_obj.is_track == 0)[0]
    idx = [i for i in temp if (event_obj.reco_num_hits_w[i] >= min_hits) and (event_obj.reco_num_hits_u[i] >= min_hits) and (event_obj.reco_num_hits_v[i] >= min_hits) and (event_obj.purity[i] >= purity)]    
    return idx