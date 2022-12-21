import numpy as np
import variables
from csvtools.csvwrite import extract_from_uproot, array_to_csv
from uproot_io import Events

"""
Script for extracting features from a .root file through uproot_io.py
Creates a csv file in the current working directory with a specified name.
"""

functions = [getattr(variables, func) for func in variables.__all__]
headers = []
for funcname in variables.__all__:
    for direction in ["u", "v", "w"]:
        headers.append(f"{funcname}_{direction}")


event = Events("c:/users/murta/documents/project_22/datasets/PandoraRecoFile_1.root")
default_headers = ["index", "n_hits_total", "purity", "completeness"]
default_data = np.array([range(len(event.reco_particle_index)),
                        event.reco_num_hits_total,
                        event.purity,
                        event.completeness]).transpose()

array_to_csv("PandoraFeatures_1.csv", default_data, default_headers)

features = extract_from_uproot(event, functions)
array_to_csv("PandoraFeatures_1.csv", features, headers)
array_to_csv("PandoraFeatures_1.csv", event.is_track, "target", rowvars=False)
