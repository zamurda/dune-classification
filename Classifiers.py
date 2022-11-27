from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from feature_extraction import conf

class IClassifier(ABC):
    """
    enforce implementation of classifier methods
    """
    def __init__(self):
        raise NotImplementedError
        
    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError
    

class ProjectiveLikelihood:
    """
    Implements Classifier interface.
    Creates a projective likelihood classifier which separates between trks (sig) and shws (bkg)
    """
    def __init__(self, features: np.ndarray, target: list, tt_split=0.75, seeded=False):
        #psuedocount
        self.pc = 1.0
        if isinstance(features, np.ndarray):
            self.features = features
            self.target = target
            
            #train-test split the data
            vectors = features.transpose()
            self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(vectors,
                                                                                    target,
                                                                                    test_size=(1-tt_split))
            
            #prior probabilities
            self.pr_sig = len(np.where(target==1)[0])/len(self.X_train)
            self.pr_bkg = len(np.where(target==0)[0])/len(self.X_train)
            
        else:
            raise TypeError(f"Expected features in np.ndarray, received {type(features)}")
    
    def train(self, n_bins: int, dinfo=True):
        #create PDF histograms for sig and bkg separately
        sig_pdfs = np.array([])
        sig_bin_edges = np.array([])
        bkg_pdfs = np.array([])
        bkg_bin_edges = np.array([])
        for metric in self.X_train.transpose():
            s_hist, s_widths = np.histogram(metric[np.where(self.y_train == 1)], bins=n_bins, density=True)
            b_hist, b_widths = np.histogram(metric[np.where(self.y_train == 0)], bins=n_bins, density=True)
            sig_pdfs = np.append(sig_pdfs, s_hist)
            sig_bin_edges = np.append(sig_bin_edges, s_widths)
            bkg_pdfs = np.append(bkg_pdfs, b_hist)
            bkg_bin_edges = np.append(bkg_bin_edges, b_widths)
            
        self.sig_pdfs = sig_pdfs.reshape(np.shape(self.features))
        self.sig_bin_edges = sig_bin_edges.reshape(np.shape(self.features)[0], n_bins)
        self.bkg_pdfs = sig_pdfs.reshape(np.shape(self.features))
        self.bkg_bin_edges = sig_bin_edges.reshape(np.shape(self.features)[0], n_bins)
        
        #add psuedocount?
        
        #return cov matrix
        if dinfo:
            return np.cov(self.X_train, rowvar=False)
            

    def test(self, full=False, plot=True):
        """
        --  full returns confusion matrix
        --  plot=True plots the likelihood ratio histogram
        """
        self.cut = 0.5
        
        #likelihood ratio
        l_s = compute_likelihoods(self.sig_pdfs, self.sig_bin_edges, self.X_test.transpose())
        l_b = compute_likelihoods(self.bkg_pdfs, self.bkg_bin_edges, self.X_test.transpose())
        self.L_sig = (self.pr_sig*l_s)/(self.pr_sig*l_s+self.pr_bkg*l_b)
        
        #plot likelihood distribution
        if plot:
            plt.hist(self.L_sig)
            plt.show()
        
        #test model- L_sig>cut -> sig
        pred = np.ones_like(self.L_sig)
        pred[np.where(self.L_sig>self.cut)] = 1
        if full:
            return conf(pred, self.y_test)
    
    def predict(self, features: np.ndarray, rowvar=True):
        """
        --  rowvar=True means features are passed as rows
        """
        if rowvar:
            features = features
        else:
            features = features.transpose()
        
        l_s = compute_likelihoods(self.sig_pdfs, self.sig_bin_edges, features)
        l_b = compute_likelihoods(self.bkg_pdfs, self.bkg_bin_edges, features)
        L_sig = (self.pr_sig*l_s)/(self.pr_sig*l_s+self.pr_bkg*l_b)
        
        p = np.ones_like(L_sig)
        p[np.where(L_sig>self.cut)] = 1
        
        return p

    def plot_roc(self):
        pass

########################################################################################################
def compute_likelihoods(pdfs, bins, features):
    '''
    find multinomial likelihood array given an array of pdfs and array of features.
    '''
    l = np.ones_like(pdfs[0])
    i=0
    for pdf,feature in zip(pdf,features):
        bin_num = np.digitize(feature, bins[i])
        l = l * pdf[bins[i][bin_num]]
        i += 1
    return l