import numpy as np
import pandas as pd
from dctools.exceptions import FeatureExtractionError
import time
from .helpers import check_unique_headers, file_in_cwd

"""
Functions to assist feature extraction and writing to csv
"""


def extract_from_uproot(event, funcs, limit=None):
    """
    Takes in an event, and list of strings containing function objects
    Applies the functions to a numpy ndarray.

    returns np.ndarray
    """
    start = time.perf_counter()
    pnum = range(len(event.reco_particle_index)) if limit is None else range(limit)
    if isinstance(funcs, list):
        for func in funcs:
            if not callable(func):
                raise TypeError("Input function objects")
            else:
                print(f"Attempting to extract {len(funcs)} features for {len(pnum)} particles")
                features = np.array([])
                for func in funcs:
                    for direction in ["u", "v", "w"]:
                        for i in pnum:
                            t_start = time.perf_counter_ns()
                            try:
                                result = func(event, i, direction)
                                features = np.append(features, result)
                                print(f"direction {direction}, particle {i}: {func} = {result}")
                            except FeatureExtractionError:
                                features = np.append(features, -1)
                                print(f"Error occurred in feature {func}, direction {direction} for particle {i}.")
                            t_stop = time.perf_counter_ns()
                            print(f"last process took: {t_stop-t_start} ns")
        features = features.reshape(len(pnum), len(funcs)*3)

    elif callable(funcs):
        pnum = range(len(event.reco_particle_index))
        features = np.array([])
        for direction in ["u", "v", "w"]:
            for i in pnum:
                t_start = time.perf_counter_ns()
                try:
                    result = funcs(event, i, direction)
                    features = np.append(features, result)
                    print(f"direction {direction}, particle {i}: {funcs} = {result}")
                except FeatureExtractionError:
                    features = np.append(features, -1)
                    print(f"Error occurred in feature {funcs}, direction {direction} for particle {i}.")
                t_stop = time.perf_counter_ns()
                print(f"last process took: {t_stop - t_start} ns")
        features = features.reshape(len(pnum), 3)

    else:
        raise TypeError("extract_from_uproot requires a function object or list of function objects")
    end = time.perf_counter()
    print(f"Total time elapsed is {end-start} s")
    return features


def array_to_csv(filename:str, data:np.ndarray, headers:list, rowvars=True):
    """
    filename:str:       filename ending in .csv where the array should be saved
    data:np.ndarray:    data which is to be saved as a csv
    headers:list:       names of column headers
    rowvars:bool:       True if each feature vector in data is a row, false if not
    """
    data = data if rowvars else data.transpose()
    if not file_in_cwd(filename):
        if check_unique_headers(headers):
            df_headers = headers if isinstance(headers, list) else [headers]
            df = pd.DataFrame(data, columns=df_headers)
            df.to_csv(filename, index=False)
        else:
            raise ValueError("Enter unique headers, or check the same feature isn't appearing twice.")

    else:
        (append_to_df(filename, data, headers, rowvars=rowvars)).to_csv(filename, index=False)


def append_to_df(filename:str, data:np.ndarray, headers:list, rowvars=True):
    """
    Returns a new dataframe with added columns.
    Dataframe has unique headers to avoid overwriting.
    """
    data = data.transpose() if rowvars else data
    df = pd.read_csv(filename)
    if check_unique_headers(df.columns.values.tolist(), new_headers=headers):
        new_dict = {}
        if len(np.shape(data)) == 1:
            new_dict[headers] = data
        else:
            for i, name in enumerate(headers):
                new_dict[name] = data[i]
        return df.assign(**new_dict)

    else:
        raise ValueError("Enter unique headers")
