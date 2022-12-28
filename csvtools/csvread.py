from csvtools.helpers import file_in_dir
import pandas as pd
from numpy import int64


class LoadFeatures:
    """
    Helper class to load features from csv files.
    
    csv files must follow the outlined schema:
        - must have columns for n_hits_total, purity, completeness
        - must have a colum named 'target'
        - columns containing features must end in 'u', 'v' or 'w'
        
    attributes:
        self.df         (pandas.DataFrame): dataframe which csv file is read into
        self.features   (numpy.ndarray):    2d array of feature vectors
        self.targets    (numpy.ndarray):    array of targets
        
    methods:
        filter(self, reqs:dict) : returns filtered features, targets and indices
        get_pdgs(self, indices) : returns pdg codes for the specified rows in self.df   
    """
    def __init__(self, filename, clean=True):
        if not file_in_dir(filename):
            
            raise FileNotFoundError(
                f"{filename} not found"
                )
            
        else:
            
            self.df = pd.read_csv(filename)
            self._feature_headers = [i for i in self.df.columns.values.tolist() if i.endswith(("u", "v", "w"))]
            
            # filter out erroneous particles
            if clean:
                bad_rows = []
                for row in self.df[self._feature_headers].itertuples():
                    if -1.0 in row[1:]:
                        bad_rows.append(row[0])
                
                self.df.drop(
                    labels=bad_rows,
                    axis=0,
                    inplace=True
                )
                self.df.reset_index(inplace=True)
            
            self.features = (self.df[self._feature_headers]).to_numpy()
            self.targets = (self.df[["target"]]).to_numpy()
            
                
    def filter(
        self,
        reqs:dict = {
            "n_hits_total"  : 10,
            "purity"        : 0.8,
            "completeness"  : 0.9
        }
        ):
        
        unwanted = []
        minimums = list(reqs.values())
        for row in self.df[reqs.keys()].itertuples():
            if list(row[1:]) < minimums:
                unwanted.append(row[0])
        
        new = self.df.drop(
                labels=unwanted,
                axis=0,
                inplace=False
        )
        
        return (
            (new[self._feature_headers]).to_numpy(),
            (new[["target"]]).to_numpy().transpose()[0],
            (new.index.to_numpy().transpose())
        )
        
        
    def get_pdgs(self, indices:list):
        
        if "pdg" in self.df.columns.values.tolist():
            
            return (self.df["pdg"]).to_numpy()[indices]
        
        else:
            
            raise(
                ValueError("'pdg' not a column in the csv file")
            )
        
