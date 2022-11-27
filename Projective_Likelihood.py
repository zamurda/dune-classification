import numpy as np
import pandas as pd

def prob_var(tr,sh,bins):
    max_tr = max(tr)
    max_sh = max(sh)
    min_tr = min(tr)
    min_sh = min(sh)
    
    T_prob = np.array([])
    S_prob = np.array([])
    
    #if max_tr > max_sh:
    #    bins_array =np.concatenate(([min_tr],[min_sh + i*((max_tr-min_sh)/bins) for i in range(bins)],[max_tr]))
    #else:
    #    bins_array =np.concatenate(([min_tr],[min_sh + i*((max_tr-min_sh)/bins) for i in range(bins)],[max_tr,max_sh]))

    bins_array =np.concatenate(([min_tr],[min_sh + i*((max_tr-min_sh)/bins) for i in range(bins+1)]))
    tr_pdf = np.histogram(tr,bins=bins_array,density=True)[0]
    sh_pdf = np.histogram(sh,bins=bins_array,density=True)[0]

    for i in range(np.size(bins_array)-1):
        if (tr_pdf[i] == 0):
            T_prob = np.append(T_prob,0)
            S_prob = np.append(S_prob,1)
        elif (sh_pdf[i] == 0):
            T_prob = np.append(T_prob,1)
            S_prob = np.append(S_prob,0)
        else:
            T_prob = np.append(T_prob,tr_pdf[i]/(sh_pdf[i]+tr_pdf[i]))
            S_prob = np.append(S_prob,sh_pdf[i]/(sh_pdf[i]+tr_pdf[i]))

    return T_prob,S_prob,bins_array

def PL(tr,sh,file):
    #tracks = random.shuffle(tr)
    #showers = random.shuffle(sh)
    
    bins = 30
    training = 0.75
    tr_training = tr[:int(training*np.size(tr))]
    sh_training = sh[:int(training*np.size(sh))]
    tr_test =  tr[int(training*np.size(tr)):]
    sh_test = sh[int(training*np.size(sh)):]

    L_sig_tracks = np.ones_like(tr_test)
    L_bck_tracks = np.ones_like(tr_test)
    L_sig_showers = np.ones_like(sh_test)
    L_bck_showers = np.ones_like(sh_test)

    data = pd.read_csv(file)
    for col in data.columns:
        var = data[col]
        tr_training_var = [var[i] for i in tr_training]
        sh_training_var = [var[i] for i in sh_training]
        tr_test_var = [var[i] for i in tr_test]
        sh_test_var =[var[i] for i in sh_test]

        T_prob, S_prob, bins_array = prob_var(tr_training_var,sh_training_var,bins)
        tr_test_prob = np.array([])
        sh_test_prob = np.array([])

        for i in tr_test_var:
            if i < bins_array[0]:
                tr_test_prob = np.append(tr_test_prob,1)
                sh_test_prob = np.append(sh_test_prob,0)
            elif i > bins_array[-1]:
                tr_test_prob = np.append(tr_test_prob,0)
                sh_test_prob = np.append(sh_test_prob,1)
            else:
                for j in range(np.size(bins_array)-1):
                    if bins_array[j] < i < bins_array[j+1]:
                        tr_test_prob = np.append(tr_test_prob,T_prob[j])
                        sh_test_prob = np.append(sh_test_prob,S_prob[j])
        
        L_sig_tracks = L_sig_tracks*tr_test_prob
        L_bck_tracks = L_bck_tracks*sh_test_prob

        tr_test_prob = np.array([])
        sh_test_prob = np.array([])

        for i in sh_test_var:
            if i < bins_array[0]:
                tr_test_prob = np.append(tr_test_prob,1)
                sh_test_prob = np.append(sh_test_prob,0)
            elif i > bins_array[-1]:
                tr_test_prob = np.append(tr_test_prob,0)
                sh_test_prob = np.append(sh_test_prob,1)
            else:
                for j in range(np.size(bins_array)-1):
                    if bins_array[j] < i < bins_array[j+1]:
                        tr_test_prob = np.append(tr_test_prob,T_prob[j])
                        sh_test_prob = np.append(sh_test_prob,S_prob[j])

        L_sig_showers = L_sig_showers*tr_test_prob
        L_bck_showers = L_bck_showers*sh_test_prob
        
    MVA_tracks = [L_sig_tracks[i]/(L_sig_tracks[i]+L_bck_tracks[i]) for i in range(np.size(L_sig_tracks))]
    MVA_showers = [L_sig_showers[i]/(L_sig_showers[i]+L_bck_showers[i]) for i in range(np.size(L_sig_showers))]

    return MVA_tracks, MVA_showers

def MVA_var(tr,sh,file):
    bins = 30
    L_sig_tracks = np.ones_like(tr)
    L_bck_tracks = np.ones_like(tr)
    L_sig_showers = np.ones_like(sh)
    L_bck_showers = np.ones_like(sh)

    data = pd.read_csv(file)
    for col in data.columns:
        var = data[col]
        tr_var = [var[i] for i in tr]
        sh_var = [var[i] for i in sh]

        T_prob = prob_var(tr_var,sh_var,bins)[0]
        S_prob = prob_var(tr_var,sh_var,bins)[1]
        bins_array = prob_var(tr_var,sh_var,bins)[2]
        tr_test_prob = np.array([])
        sh_test_prob = np.array([])
        
        for i in tr_var:
            if i <= bins_array[0]:
                tr_test_prob = np.append(tr_test_prob,1)
                sh_test_prob = np.append(sh_test_prob,0)
            elif i >= bins_array[-1]:
                tr_test_prob = np.append(tr_test_prob,0)
                sh_test_prob = np.append(sh_test_prob,1)
            else:
                for j in range(np.size(bins_array)-1):
                    if bins_array[j] <= i < bins_array[j+1]:
                        tr_test_prob = np.append(tr_test_prob,T_prob[j])
                        sh_test_prob = np.append(sh_test_prob,S_prob[j])
 
        L_sig_tracks = L_sig_tracks*tr_test_prob
        L_bck_tracks = L_bck_tracks*sh_test_prob

        tr_test_prob = np.array([])
        sh_test_prob = np.array([])

        for i in sh_var:
            if i <= bins_array[0]:
                tr_test_prob = np.append(tr_test_prob,1)
                sh_test_prob = np.append(sh_test_prob,0)
            elif i > bins_array[-1]:
                tr_test_prob = np.append(tr_test_prob,0)
                sh_test_prob = np.append(sh_test_prob,1)
            else:
                for j in range(np.size(bins_array)-1):
                    if bins_array[j] <= i < bins_array[j+1]:
                        tr_test_prob = np.append(tr_test_prob,T_prob[j])
                        sh_test_prob = np.append(sh_test_prob,S_prob[j])

        L_sig_showers = L_sig_showers*tr_test_prob
        L_bck_showers = L_bck_showers*sh_test_prob
       
    MVA_tracks = [L_sig_tracks[i]/(L_sig_tracks[i]+L_bck_tracks[i]) for i in range(np.size(L_sig_tracks))]
    MVA_showers = [L_sig_showers[i]/(L_sig_showers[i]+L_bck_showers[i]) for i in range(np.size(L_sig_showers))]

    return MVA_tracks, MVA_showers

def MVAconf(tr,sh,divide,file):
    tracks, showers = MVA_var(tr,sh,file)
    
    showers_as_shower = np.size([i for i in showers if i > divide])
    showers_as_track = np.size(showers) - showers_as_shower
    tracks_as_shower = np.size([i for i in tracks if i < divide])
    tracks_as_track = np.size(tracks) - tracks_as_shower

    TasT = (tracks_as_track)/(tracks_as_track+tracks_as_shower)
    TasS = (tracks_as_shower)/(tracks_as_track+tracks_as_shower)
    SasT = (showers_as_track)/(showers_as_shower+showers_as_track)
    SasS = (showers_as_shower)/(showers_as_shower+showers_as_track)

    return [[ [TasT,SasT],
    [TasS,SasS] ]]

