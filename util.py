import numpy as np
import random
import scipy.stats as stats
from GPlikelihood import GPCluster
import time
import csv
from math import *

def load_simulation_data():
    # generate a data set
    num_X_clusters = 7
    num_Y_clusters = 5
    Nobs = np.array([np.random.randint(30,41) for i in range(num_X_clusters)])
    Nt1 = 6
    #a random number of realisations in each cluster
    T1 = np.random.rand(Nt1,np.sum(Nobs))
    T1.sort(0)

    #random frequency and phase for each cluster
    base_freqs = 2*np.pi + 0.3*(np.random.rand(num_X_clusters)-.5)
    base_phases = 2*np.pi*np.random.rand(num_X_clusters)
    means = np.zeros((np.sum(Nobs),Nt1))
    k=0
    for i in range(num_X_clusters):
        for j in range(Nobs[i]):
            means[k]= np.sin(base_freqs[i]*T1[:,k]+base_phases[i])
            k = k+1

    freqs = .4*np.pi + 0.01*(np.random.rand(means.shape[0])-.5)
    phases = 2*np.pi*np.random.rand(means.shape[0])
    offsets = 0.3*np.vstack([np.sin(f*T1[:,n]+p).T for f,p,n in zip(freqs,phases,range(np.sum(Nobs)))])
    X = means + offsets + np.random.randn(*means.shape)*0.05

    np.random.seed(seed=32)
    num_Y_clusters = 5

    # m0 = np.array(np.random.multivariate_normal([0,0], [[25, 0],[0, 25]],num_X_clusters))
    # m1 = np.array(np.random.multivariate_normal([0,0,0], [[2500, 0, 0],[0, 2500,0], [0,0,2500]],num_Y_clusters))
    # l1 = np.array([[100,200,100,50,150,250,150]]).T

    probs = np.array([[0.3,0.01, 0.01, 0.2, 0.5],[0.8,0.2, 0.01, 0.01, 0.01],[0.1,0.1, 0.1, 0.6, 0.1],\
                      [0.01,0.1, 0.01, 0.9, 0.01],[0.1,0.1, 0.4, 0.1, 0.3],[0.01,0.7, 0.01, 0.1, 0.2],\
                      [0.01,0.01, 0.01, 0.01, 1.]])

    true_X_clustering = np.array([set([]),set([]),set([]),set([]),set([]),set([]),set([])])
    true_X_cluster_assn = []
    true_Y_clustering = np.array([set([]),set([]),set([]),set([]),set([])])
    true_Y_cluster_assn = []
    k = 0
    for i in range(num_X_clusters):
        for j in range(Nobs[i]):
            cluster = sample_multinomial(probs[i])
            true_X_clustering[i].add(k)
            true_X_cluster_assn.append(i)
            true_Y_clustering[cluster].add(k)
            true_Y_cluster_assn.append(cluster)
            k = k+1

    l2 = np.array([len(true_Y_clustering[i]) for i in range(num_Y_clusters)])

    Nt2 = 6
    #a random number of realisations in each cluster
    T2 = np.random.rand(Nt2,np.sum(Nobs))
    T2.sort(0)
    base_freqs = 2*np.pi + 0.3*(np.random.rand(num_Y_clusters)-.5)
    base_phases = 2*np.pi*np.random.rand(num_Y_clusters)

    k = 0
    Y = np.zeros((np.sum(Nobs),Nt2))
    for i in true_Y_cluster_assn:
        f = .4*np.pi + 0.01*(np.random.rand() - 0.5)
        p = 2*np.pi*np.random.rand()
        offset = np.sin(f*T2[:,n]+p).T
        Y[k] = np.sin(base_freqs[i]*T2[:,k]+base_phases[i]) +  offset + np.random.randn(*(1,Nt2))*0.05
        k = k+1

    data = np.array([[X,T1],[Y,T2]])
    return data

def load_int_data():
    """
    load internalizing behaviours and methylation data w/ no missing values
    separate training and test sets
    return format:
    Data: data[0,0] = Nxd
    data[1,0] = Nxt
    """
    methyl = []
    with open('./data/gene-level-methyl-95inds-mean-MAVANcandidates.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            methyl.append(np.array(row))
    methyl = np.array(methyl)
    methyl_id = methyl[0,1:]
    methyl_genes = methyl[1:,0]
    methyl_mean = methyl[1:,1:].astype("double").T
    internalizing = []
    with open('./data/internalizing-nomissing-zvals-ts.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            internalizing.append(np.array(row))
    internalizing = np.array(internalizing)
    internalizing_id = internalizing[1:,0]
    internalizing_z = internalizing[1:,5:].astype("double")
    internalizing_raw = internalizing[1:,1:5].astype("double")
    T = np.zeros(internalizing_z.shape)
    T[:,0] = 0
    T[:,1] = 1
    T[:,2] = 2
    T[:,3] = 3
    return np.array([[methyl_mean,methyl_genes],[internalizing_z,T.T]])

def load_int_probe_data():
    methyl = []
    with open('./data/gene-level-methyl-95inds-mean-MAVANcandidates.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            methyl.append(np.array(row))
    methyl = np.array(methyl)
    non_missingid = methyl[0,1:]

    methyl_wm = []
    with open('./data/gene-level-methyl-153inds-mean-MAVANcandidates.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            methyl_wm.append(np.array(row))
    methyl_wm = np.array(methyl_wm)
    missingid = methyl_wm[0,1:]
    i = 0
    no_missing_indx = []
    for j in range(missingid.shape[0]): # TODO: remove constant
        if missingid[j] == non_missingid[i]:
            no_missing_indx.append(j)
            if i < 94 : #TODO: remove constant
                i = i+1
    probes = []
    with open('./data/body-probes-153inds-MAVAN.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            probes.append(np.array(row))
    probes = np.array(probes)
    probes = probes[no_missing_indx,:].astype("double")
    internalizing = []
    with open('./data/internalizing-nomissing-zvals-ts.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            internalizing.append(np.array(row))
    internalizing = np.array(internalizing)
    internalizing_id = internalizing[1:,0]
    internalizing_z = internalizing[1:,5:].astype("double")
    internalizing_raw = internalizing[1:,1:5].astype("double")
    T = np.zeros(internalizing_z.shape)
    T[:,0] = 0
    T[:,1] = 1
    T[:,2] = 2
    T[:,3] = 3
    return np.array([[probes,[0]],[internalizing_z,T.T]])

def load_ext_data():
    """
    load externalizing behaviours and methylation data w/ no missing values
    separate training and test sets
    return format:
    Data:
    """
    methyl = []
    with open('./data/gene-level-methyl-95inds-mean-MAVANcandidates.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            methyl.append(np.array(row))
    methyl = np.array(methyl)
    methyl_id = methyl[0,1:]
    methyl_genes = methyl[1:,0]
    methyl_mean = methyl[1:,1:].astype("double")
    externalizing = []
    with open('./data/externalizing-nomissing-zvals-ts.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            externalizing.append(np.array(row))
    externalizing = np.array(externalizing)
    externalizing_id = externalizing[1:,0]
    externalizing_z = externalizing[1:,5:].astype("double")
    externalizing_raw = externalizing[1:,1:5].astype("double")
    return 1

def sample_multinomial(weights, sum_weights=None):
    """
    Sample from a multinomials with the given weights
    weights: n x 1 weights for n classes
    returns: c \belongs {0,...,n-1}
    """

    if sum_weights is None:
        sum_weights = np.sum(weights)
    p = random.uniform(0, sum_weights)
    sum_roulette = 0
    for i, weight in enumerate(weights):
        if weight < 0:
            return -1
        sum_roulette = sum_roulette + weight
        if (p < sum_roulette):
            return i
    return -1

def rbf_kern(t1,t2, l,v):
    return v*np.exp(-0.5 * (t1-t2)**2 / l**2)

def multivariate_t_distribution(x,mu,Sigma,df,d):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    '''
    Num = gamma(1. * (d+df)/2)

    det = np.linalg.det(Sigma)
    inv = np.linalg.inv(Sigma)
    # gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(det,1./2)
    Denom = ( gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(det,1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu).T,inv), (x - mu))[0,0],1.* (d+df)/2))
    d = 1. * Num / Denom
    return d
