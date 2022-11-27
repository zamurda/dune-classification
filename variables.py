from uproot_io import Events, View
import numpy as np

particle_id_dict = {-2212:"ANTIPROTON", -321:"KAON -", -211:"PION -", -13:"MUON -", -11:"ELECTRON", 0:"NO BEST MATCH", 11:"POSTIRON", 13:"MUON +", 22:"GAMMA", 211:"PION +", 321:"KAON +", 2212:"PROTON", 3112:"SIGMA -", 3222:"SIGMA +"}

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

def particle_data(event_obj,num_particle,direction):
    hits_x_direction = np.array([])
    hits_x = np.array([])

    if direction.lower() == "u":
        for i in range(event_obj.reco_num_hits_u[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_u[num_particle][i])
            hits_x_direction = np.append(hits_x_direction, event_obj.reco_hits_x_u[num_particle][i])
       
    if direction.lower() == 'v':
        for i in range(event_obj.reco_num_hits_v[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_v[num_particle][i])
            hits_x_direction = np.append(hits_x_direction, event_obj.reco_hits_x_v[num_particle][i])
        
    if direction.lower() == "w":
        for i in range(event_obj.reco_num_hits_w[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_w[num_particle][i])
            hits_x_direction = np.append(hits_x_direction, event_obj.reco_hits_x_w[num_particle][i])
    
    hits_bf = np.polyfit(hits_x,hits_x_direction,1)
    hits_x_direction_fit= [i * hits_bf[0] + hits_bf[1] for i in hits_x]
    residuals = np.subtract(hits_x_direction,hits_x_direction_fit)
    length = ((hits_x[0]-hits_x[-1])**2 + (hits_x_direction_fit[0]-hits_x_direction_fit[-1])**2)**0.5
    return residuals, length

def particle_length(event_obj,num_particle,direction):
    length = particle_data(event_obj,num_particle,direction)[1]
    return length

def particle_sinuousity(event_obj,num_particle,direction):
    path_length = 0

    if direction.lower() == "u":
        for i in range(event_obj.reco_num_hits_u[num_particle]):
            dx = event_obj.reco_hits_u[num_particle][i] - event_obj.reco_hits_u[num_particle][i-1]
            dy = event_obj.reco_hits_x_u[num_particle][i] - event_obj.reco_hits_x_u[num_particle][i-1]
            path_length += (dx**2+dy**2)**0.5
       
    if direction.lower() == 'v':
        for i in range(event_obj.reco_num_hits_v[num_particle]):
            dx = event_obj.reco_hits_v[num_particle][i] - event_obj.reco_hits_v[num_particle][i-1]
            dy = event_obj.reco_hits_x_v[num_particle][i] - event_obj.reco_hits_x_v[num_particle][i-1]
            path_length += (dx**2+dy**2)**0.5
    
    if direction.lower() == "w":
        for i in range(event_obj.reco_num_hits_w[num_particle]):
            dx = event_obj.reco_hits_w[num_particle][i] - event_obj.reco_hits_w[num_particle][i-1]
            dy = event_obj.reco_hits_x_w[num_particle][i] - event_obj.reco_hits_x_w[num_particle][i-1]
            path_length += (dx**2+dy**2)**0.5

    length = particle_data(event_obj,num_particle,direction)[1]
    return path_length/length

def particle_rms(event_obj,num_particle,direction):
    residuals = particle_data(event_obj,num_particle,direction)[0]
    rms = np.std(residuals)
    return rms
