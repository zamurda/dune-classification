from uproot_io import Events, View
import numpy as np
import csv
import time
from variables import *

import warnings
warnings.filterwarnings("ignore")

filename = r"C:\Users\red20\Documents\Physics_Project\projects\CheatedRecoFile_1.root"
events = Events(filename)

savefile = r"C:\Users\red20\Documents\Physics_Project\data.csv"
particles = np.size(events.mc_pdg)
start = time.time()
file = open(savefile,'w',newline='', encoding='utf-8')
writer = csv.writer(file)
#header = ['rms_u','rms_v','rms_w','sinuosity_u','sinuosity_v','sinuosity_w','path_length_w','path_length_v','path_length_w','num_hits_u','num_hits_v','num_hits_w']
header = ['dedx_ratio_u','dedx_ratio_v','dedx_ratio_w']
writer.writerow(header)

for i in range(particles):
    row = np.array([])
    try:
        x = binned_energy_ratio(events,i,"u")
        row  = np.append(row,x)
    except ValueError or TypeError:
        row = np.append(row,0)
    except IndexError:
        row = np.append(row,-1)

    try:
        x = binned_energy_ratio(events,i,"v")
        row  = np.append(row,x)
    except ValueError or TypeError:
        row = np.append(row,0)
    except IndexError:
        row = np.append(row,-1)

    try:
        x = binned_energy_ratio(events,i,"w")
        row  = np.append(row,x)
    except ValueError or TypeError:
        row = np.append(row,0)
    except IndexError:
        row = np.append(row,-1)

    writer.writerow(row)

    if i%100 == 0:
        elapsed = time.time()
        print(i,(start - elapsed))




for i in range(particles):
    break
    row = np.array([])
    try:
        x = particle_rms(events,i,'u')
        row = np.append(row,x)
    except TypeError:
        row = np.append(row,-1)
    
    try:
        x = particle_rms(events,i,'v')
        row = np.append(row,x)
    except TypeError:
        row = np.append(row,-1)
    
    try:
        x = particle_rms(events,i,'w')
        row = np.append(row,x)
    except TypeError:
        row = np.append(row,-1)
    
    try:
        x = particle_sinuousity(events,i,'u')
        row = np.append(row,x)
    except TypeError or ZeroDivisionError:
        row = np.append(row,-1)
    
    try:
        x = particle_sinuousity(events,i,'v')
        row = np.append(row,x)
    except TypeError or ZeroDivisionError:
        row = np.append(row,-1)
    
    try:
        x = particle_sinuousity(events,i,'w')
        row = np.append(row,x)
    except TypeError or ZeroDivisionError:
        row = np.append(row,-1)

    try:
        x = particle_length(events,i,'u')
        row = np.append(row,x)
    except TypeError:
        row = np.append(row,-1)

    try:
        x = particle_length(events,i,'v')
        row = np.append(row,x)
    except TypeError:
        row = np.append(row,-1)
    
    try:
        x = particle_length(events,i,'w')
        row = np.append(row,x)
    except TypeError:
        row = np.append(row,-1)
    row = np.append(row,events.reco_num_hits_u[i])
    row = np.append(row,events.reco_num_hits_v[i])
    row = np.append(row,events.reco_num_hits_w[i])
    
    writer.writerow(row)

    if i%100 == 0:
        elapsed = time.time()
        print(i,(start - elapsed))

file.close()