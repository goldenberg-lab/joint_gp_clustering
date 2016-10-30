import numpy as np
import random
import scipy.stats as stats
from GPlikelihood import GPCluster
import time
import csv
from math import *

# for simulation data
def f0(x):
    return np.sin(x*np.pi/1.5)  + np.random.normal(0,0.2)
def f1(x):
    return np.log(x+1) + np.random.normal(0,0.2) - 0.5
def f2(x):
    if(x>3):
        return np.random.normal(0,0.3) -1
    else:
        return f0(x) + np.random.normal(0,0.2)

def f3(x):
    return 0 + np.random.normal(0,0.2)
def f4(x):
    return -1*(np.sqrt(abs(x))) + 1 + np.random.normal(0,0.3)

def load_simulation_data():
    num_X_clusters = 6
    num_Y_clusters = 5
    f = np.array([f0,f1,f2,f3,f4])
    Nobs = np.array([np.random.randint(10,21) for i in range(num_X_clusters)])
    Nt = 5
    
    true_c1 = np.array([set([]) for _ in range(num_X_clusters)])
    true_ca1 = []
    true_c2 = np.array([set([]) for _ in range(num_Y_clusters)])
    true_ca2 = []
    k = 0
    alpha = np.array([[10,0.5,0.5,0.5,0.5],[0.5,10,0.5,0.5,0.5], \
                    [0.5,0.5,10,0.5,0.5],[0.5,0.5,0.5,10,0.5],\
                    [0.5,0.5,0.5,0.5,10],[1,1,1,1,1]])
    for i in range(num_X_clusters):
        for j in range(Nobs[i]):
            cluster = sample_multinomial(np.random.dirichlet(alpha[i]))
            true_c1[i].add(k)
            true_ca1.append(i)
            true_c2[cluster].add(k)
            true_ca2.append(cluster)
            k = k+1

    means = np.random.normal(0,10, (num_X_clusters,3))
    X = np.array([np.random.multivariate_normal(means[true_ca1[i]], 5*np.eye(3)) for i in range(np.sum(Nobs))])

    T = np.zeros((Nt,np.sum(Nobs)))
    for i in range(Nt):
        T[i,:] = i

    Y = np.zeros((np.sum(Nobs),Nt))
    k = 0
    for i in true_ca2:
        Y[k,:] = [f[i](x) for x in range(Nt)]
        k = k+1

    data = np.array([[X,None],[Y,T]])
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
    T[:,1] = 6
    T[:,2] = 30
    T[:,3] = 42
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

def load_ext_probe_data():
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
    with open('./data/externalizing-nomissing-zvals-ts.txt') as csvfile:
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
