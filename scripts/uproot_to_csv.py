import os
import numpy as np
import variables
from uproot_io import Events
from csvtools.csvwrite import (
    extract_from_uproot,
    array_to_csv,
    merge_csvs
)


"""
Script for extracting features from a .root file through uproot_io.py
Creates a csv file in the current working directory with a specified name.
"""

fname = "testfeatures.csv"
functions = [getattr(variables, func) for func in variables.__all__]
default_headers = ["n_hits_total", "purity", "completeness", "pdg"]
headers = []
for funcname in variables.__all__:
    for direction in ["u", "v", "w"]:
        headers.append(f"{funcname}_{direction}")

curr_path = os.path.dirname(__file__)
rawdata_path = os.path.relpath(r"..\\rawdata")

def extract():
    
    for i in range(1,11):
        fname = f"PandoraRecoFile_{i}.root"
        product_name = f"PandoraFeatures_{i}.csv"
        event = Events(f"{rawdata_path}\\{fname}")
        
        default_data = np.array([
                        event.reco_num_hits_total,
                        event.purity,
                        event.completeness,
                        event.mc_pdg
                        ]).transpose()
        
        array_to_csv(product_name, default_data, default_headers)
        features = extract_from_uproot(event, functions, limit=None)
        array_to_csv(product_name, features, headers)
        array_to_csv(product_name, event.is_track, "target", rowvars=False)
        
filenames = [f"PandoraFeatures_{i}.csv" for i in range(1,11)]
full_headers = default_headers + headers + ["target"]

if __name__ == "__main__":
    
    extract()
    merge_csvs(filenames, full_headers, "FullPandoraFeatures.csv")
