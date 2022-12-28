import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ..data.filters import idx_in_event, pdg_in_event

particle_id_dict = {-2212:"ANTIPROTON", -321:"KAON -", -211:"PION -", -13:"MUON -", -11:"ELECTRON", 0:"NO BEST MATCH", 11:"POSTIRON", 13:"MUON +", 22:"GAMMA", 211:"PION +", 321:"KAON +", 2212:"PROTON", 3112:"SIGMA -", 3222:"SIGMA +"}


__all__ = [
    "plot_hist",
    "plot_particle",
    "plot_event",
    "plot_particle_adcs"
]


def plot_hist(A,B,n_bins,z_score):
    max_A = np.mean(A)+(z_score)*np.std(A)
    max_B = np.mean(B)+(z_score)*np.std(B)
    ma = max(max_A,max_B)
    mi = min(min(A),min(B))

    if 0 < mi < 1:
        mi = 0
    bins_arr = [mi + i*((ma-mi)/n_bins) for i in range(n_bins+1)]

    plt.hist(A,range=(min(A),max_A),histtype='step',bins=bins_arr,density=True)#,weights=(np.ones_like(A)/np.size(A)))
    plt.hist(B,range=(min(B),max_B),histtype='step',bins=bins_arr,density=True)#,weights=(np.ones_like(B)/np.size(B)))

def plot_particle(event_obj,num_particle,direction):
    hits_x_dir=np.array([])
    hits_x=np.array([])
    hits_adcs=np.array([])

    if direction.lower() == "w":
        for i in range(event_obj.reco_num_hits_w[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_w[num_particle][i])
            hits_x_dir = np.append(hits_x_dir, event_obj.reco_hits_x_w[num_particle][i])
            hits_adcs = np.append(hits_adcs, event_obj.reco_adcs_w[num_particle][i])

    if direction.lower() == "v":
        for i in range(event_obj.reco_num_hits_v[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_v[num_particle][i])
            hits_x_dir = np.append(hits_x_dir, event_obj.reco_hits_x_v[num_particle][i])
            hits_adcs = np.append(hits_adcs, event_obj.reco_adcs_v[num_particle][i])
    
    if direction.lower() == "u":
        for i in range(event_obj.reco_num_hits_u[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_u[num_particle][i])
            hits_x_dir = np.append(hits_x_dir, event_obj.reco_hits_x_u[num_particle][i])
            hits_adcs = np.append(hits_adcs, event_obj.reco_adcs_u[num_particle][i])
    
    hits_bf = np.polyfit(hits_x,hits_x_dir,1)
    plt.scatter(hits_x,hits_x_dir,s=8,c=hits_adcs,cmap='hot')
    plt.plot([hits_x[0],hits_x[-1]],[hits_x[0]*hits_bf[0]+hits_bf[1],hits_x[-1]*hits_bf[0]+hits_bf[1]])
    plt.text(hits_x[-1],hits_x[-1]*hits_bf[0]+hits_bf[1],str(num_particle),c="blue")

def plot_particle_adcs(event_obj,num_particle,direction):
    hits_x=np.array([])
    hits_adcs=np.array([])

    if direction.lower() == "w":
        for i in range(event_obj.reco_num_hits_w[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_w[num_particle][i])
            hits_adcs = np.append(hits_adcs, event_obj.reco_adcs_w[num_particle][i])
   
    if direction.lower() == "v":
        for i in range(event_obj.reco_num_hits_v[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_v[num_particle][i])
            hits_adcs = np.append(hits_adcs, event_obj.reco_adcs_v[num_particle][i])
    
    if direction.lower() == "u":
        for i in range(event_obj.reco_num_hits_u[num_particle]):
            hits_x = np.append(hits_x, event_obj.reco_hits_u[num_particle][i])
            hits_adcs = np.append(hits_adcs, event_obj.reco_adcs_u[num_particle][i])
    
    plt.scatter(hits_x,hits_adcs,s=8)
    


def plot_event(event_obj,number_event,direction,min_hits,purity, *args):
    temp = idx_in_event(event_obj,number_event,direction,min_hits,purity)
    pdg = pdg_in_event(event_obj,number_event,direction,min_hits,purity)[0]
    idx = [temp[i] for i in range(np.size(temp)) if pdg[i] in np.array(args)]
    if np.size(idx) != 0:
        for i in idx:
            plot_particle(event_obj,i,direction)
    else:
        for i in temp:
            plot_particle(event_obj,i,direction)
    print(np.dstack((idx_in_event(event_obj,number_event,direction,min_hits,purity),pdg_in_event(event_obj,number_event,direction,min_hits,purity)[0],pdg_in_event(event_obj,number_event,direction,min_hits,purity)[1])))