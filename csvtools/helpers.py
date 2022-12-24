import os
from os.path import isfile, join
from collections import Counter as ctr
import ntpath

DEFAULT_RELATIVE_PATH = r"../datasets"


def file_in_dir(filename:str, path=None) -> bool:
    
    if path is not None:
        path = path
        
    else:
        path, filename = sep_file_from_path(filename) 
        
    files = [file for file in os.listdir(path) if isfile(join(path, file))]
    return filename in files


def file_in_cwd(filename:str) -> bool:
    """
    True if the specified filename is in the cwd
    """
    files = [file for file in os.listdir(os.getcwd()) if isfile(join(os.getcwd(), file))]
    return filename in files


def check_unique_headers(existing_headers:list, new_headers:list=None) -> bool:
    """
    True if all headers are unique
    """
    if new_headers is not None:
        
        if isinstance(new_headers, list):
            
            return (
                (ctr(list(set(existing_headers))) == ctr(existing_headers))
                and (ctr(list(set(new_headers))) == ctr(new_headers))
                and not (any([header in existing_headers for header in new_headers]))
                )
            
        else:
            
            return not (new_headers in existing_headers)
    else:
        return ctr(list(set(existing_headers))) == ctr(existing_headers) if isinstance(existing_headers, list) else True


def sep_file_from_path(path:str) -> str:
    """
    Returns path and filename 
    """
    top, bottom = ntpath.split(path)
    return (
        top,
        bottom or ntpath.basename(top)
    )
