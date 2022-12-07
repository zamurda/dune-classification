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
    def __init__(self, features: np.ndarray, target: list, tt_split=0.75, rowvars = True, seed=False):
        #psuedocount
        self.pc = 1.0
        if isinstance(features, np.ndarray):
            self.features = features
            self.target = target
            
            #train-test split the data
            vectors = features.transpose()
            if not seed:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(vectors,
                                                                                        target,
                                                                                        test_size=(1-tt_split))
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(vectors,
                                                                                        target,
                                                                                        test_size=(1-tt_split),
                                                                                        random_state=int(seed))
            
            #prior probabilities
            n_particles = len(features[0])
            self.pr_sig = len(np.where(target==1)[0])/n_particles
            self.pr_bkg = len(np.where(target==0)[0])/n_particles
            
        else:
            raise TypeError(f"Expected features in np.ndarray, received {type(features)}")
    
    def train(self, n_bins: int, dinfo=True):
        #create PDF histograms for sig and bkg separately
        sig_pdfs = np.array([])
        sig_bin_edges = np.array([])
        bkg_pdfs = np.array([])
        bkg_bin_edges = np.array([])
        for metric in self.X_train.transpose():
            s_hist, s_edges = np.histogram(metric[np.where(self.y_train == 1)], bins=n_bins, density=True)
            b_hist, b_edges = np.histogram(metric[np.where(self.y_train == 0)], bins=n_bins, density=True)
            sig_pdfs = np.append(sig_pdfs, s_hist)
            sig_bin_edges = np.append(sig_bin_edges, s_edges)
            bkg_pdfs = np.append(bkg_pdfs, b_hist)
            bkg_bin_edges = np.append(bkg_bin_edges, b_edges)
            
        self.sig_pdfs = sig_pdfs.reshape(np.shape(self.X_train)[1], n_bins)
        self.sig_bin_edges = sig_bin_edges.reshape(np.shape(self.X_train)[1], n_bins+1)
        self.bkg_pdfs = bkg_pdfs.reshape(np.shape(self.X_train)[1], n_bins)
        self.bkg_bin_edges = bkg_bin_edges.reshape(np.shape(self.X_train)[1], n_bins+1)
        
        #add psuedocount?
        
        #return cov matrix
        if dinfo:
            return np.corrcoef(self.X_train, rowvar=False)
            

    def test(self, plot=True, optimize=True):
        """
        --  full returns confusion matrix
        --  plot=True plots the likelihood ratio histogram
        """
        self.cut = 0.5
        
        #likelihood ratio
        l_s = compute_likelihoods(self.sig_pdfs, self.sig_bin_edges, self.X_test.transpose())
        l_b = compute_likelihoods(self.bkg_pdfs, self.bkg_bin_edges, self.X_test.transpose())
        self.L_sig = (self.pr_sig*l_s)/((self.pr_sig*l_s)+(self.pr_bkg*l_b))
        #self.L_sig = (l_s)/((l_s)+(l_b))
        
        #plot likelihood distribution
        if plot:
            sig = self.L_sig[np.where(self.y_test==1)]
            bkg = self.L_sig[np.where(self.y_test==0)]
            plt.hist(sig, 100, color="b", histtype="step", log=True)
            plt.hist(bkg, 100, color="r", histtype="step", log=True)
            plt.show()
        
        #optimize cut if needed
        if optimize:
            
            c = np.linspace(0,1,1001, endpoint=False)
            tr = np.array([])
            for cut in c:
                pred = np.ones_like(self.L_sig)
                pred[np.where(self.L_sig < cut)] = 0
                a = conf(pred, self.y_test)
                t = np.trace(a)
                tr = np.append(tr, t/2)
            self.cut = c[tr.argmax()]
            pred = np.ones_like(self.L_sig)
            pred[np.where(self.L_sig < self.cut)] = 0
        
        else:
            self.cut = 0.5
            pred = np.ones_like(self.L_sig)
            pred[np.where(self.L_sig < self.cut)] = 0
            
        return conf(pred, self.y_test, diagnostic=False)
        
    
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
        L_sig = (self.pr_sig*l_s)/((self.pr_sig*l_s)+(self.pr_bkg*l_b))
        
        p = np.ones_like(L_sig)
        p[np.where(L_sig < self.cut)] = 0
        
        return p

    def plot_roc(self):
        cuts = np.linspace(0,1,1001, endpoint=False)
        self.efficiency = np.array([])
        self.purity = np.array([])
        for curr_cut in cuts:
            temp_pred = np.zeros_like(self.L_sig)
            temp_pred[np.where(self.L_sig>=curr_cut)] = 1
            eff, pur = conf(temp_pred, self.y_test, diagnostic=True)
                
            self.efficiency = np.append(self.efficiency, eff)
            self.purity = np.append(self.purity, pur)
        plt.xlabel("Purity")
        plt.ylabel("Efficiency")
        plt.title(f"The optimal cut is at {np.around(self.cut, 2)}")
        plt.plot(self.efficiency, self.purity)
        plt.axvline(self.efficiency[np.where(cuts==self.cut)], ls="dotted", c="g")
        plt.axhline(self.purity[np.where(cuts==self.cut)], ls="dotted", c="g")
        plt.show()

########################################################################################################
def compute_likelihoods(pdfs, bins, features):
    '''
    find multinomial likelihood array given an array of pdfs and array of features.
    '''
    l = np.ones_like(features[0])
    i=0
    for pdf,feature in zip(pdfs,features):
        pdf[np.where(pdf==0.0)] = 1/10000
        bin_num = np.digitize(feature, bins[i]) - 1 #to get correct hist array index
        #edge case handling
        bin_num[np.where(bin_num == (len(bins[i])-1))] = len(bins[i])-2
        bin_num[np.where(bin_num == -1)] = 0
        l = l * pdf[bin_num]
        i += 1
    return l