import numpy as np


__all__ = [
    "create_knl",
    "peak_finder"
]


def create_knl(n_hits,s):
    kSize = lambda x: int(np.floor(6*np.log10(x+1))) if x < 200 else int(np.floor(6*np.log(x+1)))
    if n_hits <= 15:
        w = 3
    elif n_hits <= 40:
        w = int(np.floor(3*np.log10(n_hits+1)))
    else:
        w = kSize(n_hits)
    mid = w//2
    return ((1/(np.sqrt(np.pi*2)*s)) * np.exp((-np.linspace(-mid,mid,w)**2)/(2*(s**2)))), w

def peak_finder(y, threshold, influence, n_hits):
    lagSetter = lambda x: int(np.floor(4*np.log10(x+1))) if x >= 50 else int(np.floor(2*np.log10(x+1)))
    lag = lagSetter(n_hits)
    print(f"taking an avg of {lag} points")
    signal = np.zeros_like(y)
    K = np.copy(y)
    avgK = np.zeros_like(y)
    stdK = np.zeros_like(y)
    avgK[lag-1] = np.mean(K[0:lag])
    stdK[lag-1] = np.std(K[0:lag])
    
    for i in range(lag,np.size(y)):
        
        if np.abs(y[i] - avgK[i-1]) > threshold*stdK[i-1]:
            if y[i] > avgK[i-1]:
                signal[i] = 1
            else:
                signal[i] = 0 #don't care about troughs so set to 0
            K[i] = influence*y[i] + (1-influence)*y[i-1]
        else:
            signal[i] = 0
            K[i] = y[i]
        
        avgK[i] = np.mean(K[(i-lag+1):(i+1)])
        stdK[i] = np.std(K[(i-lag+1):(i+1)])
    
    return signal,avgK,stdK

def findMax(signals,x,y):
    if signals.any():
        max_peak = np.max(y[signals==1])
        return x[y==max_peak]
    else:
        return -1