import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def prob_var(tr,sh,bins):
    max_tr = max(tr)
    max_sh = max(sh)
    min_tr = min(tr)
    min_sh = min(sh)
    
    T_prob = np.array([])
    S_prob = np.array([])

    if max_sh > max_tr and min_tr < min_sh:
        bins_arr = np.concatenate(([min_tr],np.linspace(min_sh,max_tr,bins+1)))
    elif max_sh > max_tr and min_sh < min_tr:
        bins_arr = np.concatenate(([min_sh],np.linspace(min_tr,max_tr,bins+1)))
    elif max_tr > max_sh and min_tr <= min_sh:
        bins_arr = np.concatenate(([min_tr],np.linspace(min_sh,max_sh,bins+1),[max_tr]))
    elif max_tr > max_sh and min_sh <= min_tr:
        bins_arr = np.concatenate(([min_sh],np.linspace(min_tr,max_sh,bins+1),[max_tr]))
    elif min_tr == min_sh and max_sh > max_tr:
        bins_arr = np.linspace(min_sh,max_tr,bins+1)
    elif min_tr == max_sh and max_tr > max_sh:
        bins_arr = np.concatenate((np.linspace(min_sh,max_sh),[max_tr]))
    
    tr_pdf = np.histogram(tr,bins=bins_arr,density=True)[0]
    sh_pdf = np.histogram(sh,bins=bins_arr,density=True)[0]

    for i in range(np.size(bins_arr)-1):
        if (tr_pdf[i] == 0):
            T_prob = np.append(T_prob,0.001)
            S_prob = np.append(S_prob,1)
        elif (sh_pdf[i] == 0):
            T_prob = np.append(T_prob,1)
            S_prob = np.append(S_prob,0.001)
        else:
            T_prob = np.append(T_prob,tr_pdf[i]/(sh_pdf[i]+tr_pdf[i]))
            S_prob = np.append(S_prob,sh_pdf[i]/(sh_pdf[i]+tr_pdf[i]))

    return T_prob,S_prob,bins_arr

def PL(tr,sh,data,shuffle):
    if shuffle == True:
        np.random.shuffle(tr)
        np.random.shuffle(sh)
    
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

    for col in range(np.size(data[0])):
        var = data[:,col]
        tr_training_var = var[tr_training]
        sh_training_var = var[sh_training]
        tr_test_var = var[tr_test]
        sh_test_var = var[sh_test]

        T_prob, S_prob, bins_array = prob_var(tr_training_var,sh_training_var,bins)
        tr_test_prob = np.array([])
        sh_test_prob = np.array([])

        for i in tr_test_var:
            if i < bins_array[0]:
                tr_test_prob = np.append(tr_test_prob,1)
                sh_test_prob = np.append(sh_test_prob,1/np.size(tr_test_var))
            elif i >= bins_array[-1]:
                tr_test_prob = np.append(tr_test_prob,1/np.size(tr_test_var))
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

        for i in sh_test_var:
            if i < bins_array[0]:
                tr_test_prob = np.append(tr_test_prob,1)
                sh_test_prob = np.append(sh_test_prob,1/np.size(sh_test_var))
            elif i >= bins_array[-1]:
                tr_test_prob = np.append(tr_test_prob,1/np.size(sh_test_var))
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

def MVA_var(tr,sh,data):
    bins = 30
    L_sig_tracks = np.ones_like(tr)
    L_bck_tracks = np.ones_like(tr)
    L_sig_showers = np.ones_like(sh)
    L_bck_showers = np.ones_like(sh)

    rows, columns =data.shape
    for col in range(columns):
        var = data[:,col]
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

def MVAconf(tr,sh,divide,data,test):
    if test == 0:
        tr,sh = PL(tr,sh,data)
    elif test == 1:
        tr, sh = MVA_var(tr,sh,data)
    elif test != 2:
        print("Invalid test parameter:\ntest == 0 for PL on sample\ntest == 0 for full MVA on sample\ntest == 2 for inputs being a variable, file not needed")

    showers_as_shower = np.size([i for i in sh if i < divide])
    showers_as_track = np.size(sh) - showers_as_shower
    tracks_as_shower = np.size([i for i in tr if i < divide])
    tracks_as_track = np.size(tr) - tracks_as_shower

    TTasT = (tracks_as_track)/(tracks_as_track+tracks_as_shower)
    TTasS = (tracks_as_shower)/(tracks_as_track+tracks_as_shower)
    TSasT = (showers_as_track)/(showers_as_shower+showers_as_track)
    TSasS = (showers_as_shower)/(showers_as_shower+showers_as_track)

    return [ [TTasT,TTasS],
    [TSasT,TSasS] ]

def PLROC_divde(tr,sh,plot):

    divisions = 1000
    nshowers = np.size(sh)  #real negatives
    ntracks = np.size(tr)    #real postives

    if plot == True:
        efficiency = np.array([[],[]])
        purity = np.array([[],[]])
    s_ep_max,t_ep_max = 0, 0

    for divide in np.linspace(0,1,divisions+1):
        showers_as_shower = np.size([i for i in sh if i < divide]) #tn
        showers_as_track = np.size(sh) - showers_as_shower         #fn
        tracks_as_shower = np.size([i for i in tr if i < divide])   #fp
        tracks_as_track = np.size(tr) - tracks_as_shower            #tp

        try:
            t_eff =tracks_as_track/ntracks        #True postive rate tp/(tot p)
            # fnr = 1 - t_eff                       #False negative rate    1 - tp/(tot p)
            s_eff = showers_as_shower/nshowers    #True negative rate     tn/(tot n)
            # fpr = 1 - tnr                       #False negative rate    1- tn/tot n
            t_pur = (tracks_as_track)/(tracks_as_track+showers_as_track)         #postive precision
            s_pur = (showers_as_shower)/(showers_as_shower+tracks_as_shower)     #negative precision
        except ZeroDivisionError:
            t_eff = 0
            s_eff = 1
            t_pur = 1
            s_pur = 0
        
        if plot == True:
            efficiency = np.append(efficiency,[[t_eff],[s_eff]],axis = 1)
            purity = np.append(purity,[[t_pur],[s_pur]], axis = 1)
    
        t_ep = t_eff * t_pur
        s_ep = s_eff * s_pur

        if t_ep >= t_ep_max:
            t_ep_max = t_ep
            t_best_divide = divide
            t_best_eff = t_eff
            t_best_pur = t_pur
        if s_ep > s_ep_max:
            s_ep_max = s_ep
            s_best_divide = divide
            s_best_eff = s_eff
            s_best_pur = s_pur
    
    if plot == True:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,6))
        ax1.set_xlabel("Track Purity")
        ax1.set_ylabel("Track Efficiency")
        ax1.scatter(purity[0],efficiency[0],s = 8)
        ax2.set_xlabel("Shower Purity")
        ax2.set_ylabel("Shower Efficiency")
        ax2.scatter(purity[1],efficiency[1],s=8)
        print("Best track divide is at",t_best_divide,"\n best track efficiency is",t_best_eff,"best track purity is",t_best_pur)
        print("Best shower divide is at",s_best_divide,"\n best shower efficiency is",s_best_eff,"best shower purity is",s_best_pur)
        return efficiency,purity
    if plot == False:
        return t_best_divide,s_best_divide
