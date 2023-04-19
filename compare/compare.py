# other package
import numpy as np
import pickle as pkl
import shutil
import matplotlib.pyplot as plt

# MEArec
import MEArec as mr
import MEAutility as mu

# spikeinterface
import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw

# math
from sklearn.metrics import jaccard_score
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds
from math import erf
from sklearn.metrics.pairwise import cosine_similarity

####### temporal spike time/train comparison

def get_spike_time_byUnitList(sorting_algo):
    """ transfer sorting to spike time list by unit

    Args:
        sorting_algo (SortingExtractor): sorting of algo

    Returns:
        list: spike time by unit list
    """    

    st = []
    id_list = sorting_algo.get_unit_ids()
    for id in id_list:
        st_i = sorting_algo.get_all_spike_trains()[0][0][np.where(sorting_algo.get_all_spike_trains()[0][1]==id)]
        st.append(st_i)

    return st

def time2train(time,fs,N,deltaT):
    """from spike time stamp to bool spike train

    Args:
        time (list): spike time list
        fs (float): orginal sampling frequency
        N (int): number of all original sampling
        deltaT (float): temporal resolution for the output bool spike train (sec)

    Returns:
        array: (floor(N/fs/deltaT),) bool spike train
    """    

    train = np.zeros(int(np.floor(N/fs/deltaT)),dtype=bool)
    for t in time:
        temp = int(np.floor(t/fs/deltaT))
        if temp >= int(np.floor(N/fs/deltaT)):
            continue
        train[temp] = True

    return train

def time2train_byUnitList(timelist,fs,N,deltaT):
    """from spike time stamp to bool spike train, by unit list

    Args:
        time (list): list of spike time lists
        fs (float): orginal sampling frequency
        N (int): number of all original sampling
        deltaT (float): temporal resolution for the output bool spike train (sec)

    Returns:
        list: (N,floor(N/fs/deltaT)), N units' bool spike trains 
    """    

    temp = [time2train(time,fs,N,deltaT)[np.newaxis,:] for time in timelist]

    return np.concatenate(temp,axis=0)

def jaccard_score_matrix(train1,train2):
    """compute jaccard score matrix of two groups of spike trains. Jaccard score is commutative. This shows general similarity between two spike trains.
        jaccard score = TP/(TP+TN+FP)

    Args:
        train1 (ndarray): (A,T)
        train2 (ndarray): (B,T)

    Returns:
        ndarray: (A,B)
    """    

    matrix = np.zeros((train1.shape[0],train2.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i,j] = jaccard_score(train1[i,:],train2[j,:])

    return matrix

def matrix_diagmax_sort(A):
    """ sort rows and columns of A to maximize diagonal sum

    Args:
        A (ndarray): (M,N)

    Returns:
        ndarray: (M,N)
    """    

    trans = False
    if A.shape[0] > A.shape[1]:
        A = A.T
        trans = True

    answer = linear_sum_assignment(A,maximize=True)
    B = np.zeros_like(A)
    for i in range(A.shape[0]):
        B[:,i] = A[:,answer[1][i]]
    complementary = []
    for i in range(A.shape[1]):
        if i not in answer[1]:
            complementary.append(i)
    # print(complementary)
    B[:,A.shape[0]:] = A[:,complementary]

    if trans:
        B = B.T

    return B

def time_agreement_score(time1, time2, fs, tol):
    """compute spike time agreement score between two units.
    matching: exist t2 within period [t1-tol,t1+tol]

    Args:
        time1 (list): spike time
        time2 (list): spike time
        fs (float): sampling rate
        tol (float): tolerant time (sec)

    Returns:
        float: spike time agreement score
    """    

    if len(time1) == 0 or len(time2) == 0:
        return 0

    t1 = np.sort(time1)
    t2 = np.sort(time2)

    n_match = 0
    i = 0 # pointer of t1
    j = 0 # lower pointer of t2

    while i < len(t1):
        period = [t1[i]-np.ceil(tol*fs),t1[i]+np.ceil(tol*fs)]
        while j < len(t2) and t2[j] <= period[0]:
            j = j+1
        if j == len(t2):
            break
        if t2[j] < period[1]:
            n_match = n_match+1
        i = i+1


    score = n_match/(len(t1)+len(t2)-n_match)

    return score

def time_agreement_matrix(timelist1,timelist2,fs,tol):
    """compute spike time agreement matrix between two groups of units.

    Args:
        timelist1 (list): N1 list of spike time lists
        timelist2 (list): N2 list of spike time lists
        fs (float): sampling rate
        tol (float): tolerant time (sec)

    Returns:
        ndarray: agreement matrix (N1,N2)
    """    

    N1 = len(timelist1)
    N2 = len(timelist2)
    matrix = np.zeros((N1,N2))
    for i in range(N1):
        for j in range(N2):
            matrix[i,j] = time_agreement_score(timelist1[i],timelist2[j],fs,tol)
    return matrix

# ccg from pykilosort
def ccg_slow(st1, st2, nbins, tbin,T):
    # this function efficiently computes the crosscorrelogram between two sets
    # of spikes (st1, st2), with tbin length each, timelags =  plus/minus nbins
    # and then estimates how refractory the cross-correlogram is, which can be used
    # during merge decisions.

    st1 = np.sort(st1)  # makes sure spike trains are sorted in increasing order
    st2 = np.sort(st2)

    dt = nbins * tbin

    N1 = max(1, len(st1))
    N2 = max(1, len(st2))
    # T = np.concatenate((st1, st2)).max() - np.concatenate((st1, st2)).min()

    # we traverse both spike trains together, keeping track of the spikes in the first
    # spike train that are within dt of spikes in the second spike train

    ilow = 0  # lower bound index
    ihigh = 0  # higher bound index
    j = 0  # index of the considered spike

    K = np.zeros(2 * nbins + 1)

    # (DEV_NOTES) the while loop below is far too slow as is

    while j <= N2 - 1:  # traverse all spikes in the second spike train

        while (ihigh <= N1 - 1) and (st1[ihigh] < st2[j] + dt):
            ihigh += 1  # keep increasing higher bound until it's OUTSIDE of dt range

        while (ilow <= N1 - 1) and (st1[ilow] <= st2[j] - dt):
            ilow += 1  # keep increasing lower bound until it's INSIDE of dt range

        if ilow > N1 - 1:
            break  # break if we exhausted the spikes from the first spike train

        if st1[ilow] > st2[j] + dt:
            # if the lower bound is actually outside of dt range, means we overshot (there were no
            # spikes in range)
            # simply move on to next spike from second spike train
            j += 1
            continue

        for k in range(ilow, ihigh):
            # for all spikes within plus/minus dt range
            ibin = np.rint((st2[j] - st1[k]) / tbin).astype(int)  # convert ISI to integer

            K[ibin + nbins] += 1

        j += 1

    irange1 = np.concatenate((np.arange(1, nbins // 2), np.arange(3 * nbins // 2, 2 * nbins)))
    irange2 = np.arange(nbins - 50, nbins - 10)
    irange3 = np.arange(nbins + 11, nbins + 50)

    # normalize the shoulders by what's expected from the mean firing rates
    # a non-refractive poisson process should yield 1

    Q00 = np.sum(K[irange1]) / (len(irange1) * tbin * N1 * N2 / T)
    # do the same for irange 2
    Q01 = np.sum(K[irange2]) / (len(irange2) * tbin * N1 * N2 / T)
    # compare to the other shoulder
    Q01 = np.max([Q01, np.sum(K[irange3]) / (len(irange3) * tbin * N1 * N2 / T)])

    R00 = np.max([np.mean(K[irange2]), np.mean(K[irange3])])  # take the biggest shoulder
    R00 = np.max([R00, np.mean(K[irange1])])  # compare this to the asymptotic shoulder

    # test the probability that a central area in the autocorrelogram might be refractory
    # test increasingly larger areas of the central CCG

    a = K[nbins]
    K[nbins] = 0

    Qi = np.zeros(10)
    Ri = np.zeros(10)

    for i in range(1, 11):
        irange = np.arange(nbins - i, nbins + i + 1)  # for this central range of the CCG
        # compute the normalised ratio as above. this should be 1 if there is no refractoriness
        Qi0 = np.sum(K[irange]) / (2 * i * tbin * N1 * N2 / T)
        Qi[i - 1] = Qi0  # save the normalised probability

        n = np.sum(K[irange]) / 2
        lam = R00 * i

        # log(p) = log(lam) * n - lam - gammaln(n+1)

        # this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean and
        # variance that allows us to integrate the probability that we would see <N spikes in the
        # center of the cross-correlogram from a distribution with mean R00*i spikes

        p = 1 / 2 * (1 + erf((n - lam) / np.sqrt(2 * lam)))

        Ri[i - 1] = p  # keep track of p for each bin size i

    K[nbins] = a  # restore the center value of the cross-correlogram

    return K, Qi, Q00, Q01, Ri

def ccg_matrix(timelist1,timelist2,fs,nbins,tbin,N):
    """compute ccg matrix for 2 groups of spike time lists

    Args:
        timelist1 (list): list of spike time lists
        timelist2 (list): list of spike time lists
        fs (float): sampling rate
        nbins (int): number of bins to compute
        tbin (float): each bin time (sec)
        N (int): number of samples

    Returns:
        ndarray: ccg matrix
    """    


    N1 = len(timelist1)
    N2 = len(timelist2)
    matrix = np.zeros((N1,N2))

    for i,time1 in enumerate(timelist1):
        if len(time1) == 0:
            matrix[i,:] = 0
            continue
        for j,time2 in enumerate(timelist2):
            if len(time2) == 0:
                matrix[i,j] = 0
                continue
            K, Qi, Q00, Q01, Ri = ccg_slow(time1/fs,time2/fs,nbins,tbin,N/fs)
            if np.isnan(np.min(Qi/(np.max([Q00, Q01])))):
                matrix[i,j] = 0
            elif np.min(Qi/(np.max([Q00, Q01])))>=1:
                matrix[i,j] = 1
            else:
                matrix[i,j] = np.min(Qi/(np.max([Q00, Q01])))
            
    return matrix

####### template waveform comparison

def simscore_matrix(template1,template2,k=6,peak_index=32):
    """ Compute simscore based on spatiotemporal waveform similarity

    Args:
        template1 (ndarray): (N1,Ch,T)
        template2 (ndarray): (N1,Ch,T)
        k (int, optional): svd component dim to compute. Defaults to 6.
        peak_index (int, optional): index to spike peak, for determining wsign. Defaults to 32.

    Returns:
        _type_: _description_
    """    
    
    # number of templates
    N1 = template1.shape[0]
    N2 = template2.shape[0]

    templates = np.concatenate([template1,template2],axis=0)

    U = np.zeros((templates.shape[0],templates.shape[1],k))
    W = np.zeros((templates.shape[0],templates.shape[2],k))

    for i in range(templates.shape[0]):
        u, s, w = svds(templates[i,:,:], k)
        wsign = -np.sign(w[-1,peak_index])
        W[i,:,:] = (wsign*w).T
        u = wsign*u*s
        mu = np.sqrt(np.sum(np.sum(u ** 2)))
        if mu == 0:
            U[i,:,:] = 0
        else:
            U[i,:,:] = u / mu

    simScore = np.multiply((np.einsum('ab,cb->ac',U.reshape((templates.shape[0],-1)),U.reshape((templates.shape[0],-1)))), 
                        (np.einsum('ab,cb->ac',W.reshape((templates.shape[0],-1)),W.reshape((templates.shape[0],-1)))))/k

    result = simScore[:N1,N1:]
    return result

def cosine_similarity_matrix(template1,template2):
    """compute cosine similarity matrix of 2 groups of templates

    Args:
        template1 (ndarray): (N1,ch,T)
        template2 (ndarray): (N2,ch,T)

    Returns:
        ndarray: (N1,N2)
    """    

    template1_flat = template1.reshape(template1.shape[0], -1)
    template2_flat = template2.reshape(template2.shape[0], -1)
    score = cosine_similarity(template1_flat,template2_flat)
    return score