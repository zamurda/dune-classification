import numpy as np
import variables
from csvtools.csvwrite import extract_from_uproot, array_to_csv, merge_csvs
from uproot_io import Events

"""
Script for extracting features from a .root file through uproot_io.py
Creates a csv file in the current working directory with a specified name.
"""

fname = "testfeatures.csv"
functions = [getattr(variables, func) for func in variables.__all__]
headers = []
for funcname in variables.__all__:
    for direction in ["u", "v", "w"]:
        headers.append(f"{funcname}_{direction}")


event = Events("c:/users/murta/documents/project_22/datasets/PandoraRecoFile_1.root")
print(f"starting the extraction........")

default_headers = ["n_hits_total", "purity", "completeness", "pdg"]
default_data = np.array([
                        event.reco_num_hits_total[:500],
                        event.purity[:500],
                        event.completeness[:500],
                        event.mc_pdg[:500]
                        ]).transpose()

array_to_csv(fname, default_data, default_headers)

features = extract_from_uproot(event, functions, limit=500)
array_to_csv(fname, features, headers)
array_to_csv(fname, event.is_track[:500], "target", rowvars=False)
