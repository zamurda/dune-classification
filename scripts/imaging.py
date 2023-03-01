import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dctools.data import trk_lookup, shw_lookup
from uproot_io import Events

DATA_PATH = r"c:/users/murta/documents/project_22/datasets/CheatedRecoFile_1.root"
event = Events(DATA_PATH)
trk_id = trk_lookup(event)
shw_id = shw_lookup(event)
print("\n\n----------------------EVENT LOADED----------------------------------------\n\n")
TRK_ODIR = r"c:/users/murta/documents/project_22/datasets/images/trk"
for count,idx in enumerate(trk_id):
    plt.clf()
    x,y,e = event.reco_hits_x_w[idx], event.reco_hits_w[idx], event.reco_adcs_w[idx]
    plt.axis("off")
    plt.scatter(x,y,c=e,cmap="hot")
    plt.savefig(f"{TRK_ODIR}/trk_{count}.jpg", bbox_inches="tight", transparent="false", pad_inches=0)
    print(f"saved trk {count}")

SHW_ODIR = r"c:/users/murta/documents/project_22/datasets/images/shw"
for count,idx in enumerate(shw_id):
    plt.clf()
    x,y,e = event.reco_hits_x_w[idx], event.reco_hits_w[idx], event.reco_adcs_w[idx]
    plt.axis("off")
    plt.scatter(x,y,c=e,cmap="hot")
    plt.savefig(f"{SHW_ODIR}/shw_{count}.jpg", bbox_inches="tight", transparent="false", pad_inches=0)
    print(f"saved shw {count}")
