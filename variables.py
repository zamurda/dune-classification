from uproot_io import Events, View
import numpy as np
from cleaners import createKnl
from scipy.stats import binned_statistic

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

    if direction.lower() == "u":
        nhits = event_obj.reco_num_hits_u[num_particle]
        hits_x = event_obj.reco_hits_u[num_particle]
        hits_x_direction = event_obj.reco_hits_x_u[num_particle]
       
    if direction.lower() == 'v':
        nhits = event_obj.reco_num_hits_v[num_particle]
        hits_x = event_obj.reco_hits_v[num_particle]
        hits_x_direction = event_obj.reco_hits_x_v[num_particle]
        
    if direction.lower() == "w":
        nhits = event_obj.reco_num_hits_w[num_particle]
        hits_x = event_obj.reco_hits_w[num_particle]
        hits_x_direction = event_obj.reco_hits_x_w[num_particle]
    
    coeff, residual,x,y,z = np.polyfit(hits_x,hits_x_direction,1,full=True)
    length = ((hits_x[0]-hits_x[-1])**2 + (coeff[0]*hits_x[0]-coeff[0]*hits_x[-1])**2)**0.5
    rms = (residual/nhits)**0.5
    if nhits in [0,1]:
        return 0,0
    return rms, length

def particle_length(event_obj,num_particle,direction):
    length = particle_data(event_obj,num_particle,direction)[1]
    return length

def particle_sinuousity(event_obj,num_particle,direction):
    path_length = 0

    if direction.lower() == "u":
        for i in range(1,event_obj.reco_num_hits_u[num_particle]):
            dx = event_obj.reco_hits_u[num_particle][i] - event_obj.reco_hits_u[num_particle][i-1]
            dy = event_obj.reco_hits_x_u[num_particle][i] - event_obj.reco_hits_x_u[num_particle][i-1]
            path_length += (dx**2+dy**2)**0.5
       
    if direction.lower() == 'v':
        for i in range(1,event_obj.reco_num_hits_v[num_particle]):
            dx = event_obj.reco_hits_v[num_particle][i] - event_obj.reco_hits_v[num_particle][i-1]
            dy = event_obj.reco_hits_x_v[num_particle][i] - event_obj.reco_hits_x_v[num_particle][i-1]
            path_length += (dx**2+dy**2)**0.5
    
    if direction.lower() == "w":
        for i in range(1,event_obj.reco_num_hits_w[num_particle]):
            dx = event_obj.reco_hits_w[num_particle][i] - event_obj.reco_hits_w[num_particle][i-1]
            dy = event_obj.reco_hits_x_w[num_particle][i] - event_obj.reco_hits_x_w[num_particle][i-1]
            path_length += (dx**2+dy**2)**0.5

    length = particle_data(event_obj,num_particle,direction)[1]
    if length == 0:
        return 0
    return path_length/length

def particle_rms(event_obj,num_particle,direction):
    rms = particle_data(event_obj,num_particle,direction)[0]
    if np.size(rms) == 0:
        return 0
    return rms

def binned_energy_ratio(event_obj: Events, num_particle, direction: str):    
    if direction.lower() == "u":
        x = event_obj.reco_hits_x_u[num_particle]
        if x[0] > x[-1]:
            x = np.flip(x)
        y = event_obj.reco_hits_u[num_particle]
        adc = event_obj.reco_adcs_u[num_particle]         
    elif direction.lower() == "v":
        x = event_obj.reco_hits_x_v[num_particle]
        if x[0] > x[-1]:
            x = np.flip(x)
        y = event_obj.reco_hits_v[num_particle]
        adc = event_obj.reco_adcs_v[num_particle]        
    elif direction.lower() == "w":
        x = event_obj.reco_hits_x_w[num_particle]
        if x[0] > x[-1]:
            x = np.flip(x)
        y = event_obj.reco_hits_w[num_particle]
        adc = event_obj.reco_adcs_w[num_particle]    
    else:
        raise ValueError(f"expected u,v,w as direction, recieved {direction.lower()}")
        
    K,w = createKnl(np.size(x), 2)
    grad,intercept = np.polyfit(x,y,1)
    path = (grad*x+intercept)[w-1:]
    convolved = np.convolve(adc,K,"valid")
    arr = np.linspace(min(path),max(path),11)
    means,bins,binindices = binned_statistic(path,convolved,bins = [arr[0],arr[3],arr[8],arr[10]])
        
    ratio = means[0]/means[2]
    
    return ratio