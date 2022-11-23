from uproot_io import Events, View
import numpy as np
import csv
from variables import *

filename = r"C:\Users\red20\Documents\Physics_Project\projects\CheatedRecoFile_1.root"
events = Events(filename)

particles = np.size(events.mc_pdg)

savefile = r"C:\Users\red20\Documents\Physics_Project\Data.csv"

file = open(savefile,'w',newline='', encoding='utf-8')
writer = csv.writer(file)
header = ['rms_w','rms_v','rms_u']
writer.writerow(header)
for i in range(100):
    row = np.array([])
    
    for d in ['w','v','u']:
        try:
            rms = particle_rms(events,i,d)
            row = np.append(row,rms)
        except TypeError:
            row = np.append(row,-1)
    
    writer.writerow(row)

file.close()