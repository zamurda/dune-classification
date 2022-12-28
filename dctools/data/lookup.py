"""
functions for looking up pdg indexes of particles
"""
import numpy as np
import pandas as pd
from typing import Union

__all__ = [
    "count_pdg",
    "pdg_in_bins"
]


def count_pdg(df:pd.DataFrame, pdg:int):
    """
    Counts number of specified pdg codes in a dataframe
    """

    if "particle_pdg" not in df.columns.values.tolist():

        raise ValueError(
            "Expected a column named particle_pdg in the dataframe"
        )

    else:

        pdg_arr = df["particle_pdg"]

        return np.count_nonzero(pdg_arr == pdg)


def pdg_in_bins(
    pdg_arr:    list | np.ndarray,
    likelihood: list | np.ndarray,
    bin_edges:  list | np.ndarray,
    classes:    list | np.ndarray = None
    ):
    
    """
    Takes in a likelihood array and returns the frequency of each pdg in each bin, given an array of bin edges.
    
    params:
        pdg_arr     (list|ndarray): pdg codes
        likelihood  (list|ndarray): signal likelihood ratios as defined by a naive-Bayes classifier
        bin_edges   (list|ndarray): bin edges for the likelihood
        classes     (list|ndarray): specified pdg codes of interest (default=None)
    
    returns:
        results_dict    (dict)
        classes_str     (list of strings)
        
    """
    
    if classes is not None:
        classes_str = list(set([str(i) for i in classes]))
        classes_str.append("other")
        classes.append(0000)     # pdg code which doesn't exist, made to correspond with "other" class
        
    else:
        classes_str = list(set([str(i) for i in pdg_arr]))
        classes = list(set(pdg_arr))
    
    results_dict = {}
    
    for i in range(1,len(bin_edges)):
        pdg_in_bin = pdg_arr[(likelihood > bin_edges[i-1]) & (likelihood <= bin_edges[i])]
        
        temp_list = np.zeros_like(classes)
        results_dict[f"{np.around(bin_edges[i-1], 2)} < L_sig <= {np.around(bin_edges[i], 2)}"] = temp_list
        
        not_found = 0
        for pdg in pdg_in_bin:
            if pdg in classes:
                temp_list[classes == pdg] += 1
            else:
                not_found += 1
                         
        if not_found:
            print(not_found)
            temp_list[-1] = not_found
        
        
    return results_dict, classes_str
