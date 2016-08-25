import numpy as np
import random
import scipy.stats as stats
from GPlikelihood import GPCluster
import time

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
# Settings tried:
#   1:v1 = 1, l = 0.05, v2 = 1
#   1:same
#   2: 0.5,0.05,0.5
def Px_z1(data, i, clustering, cluster_assn):
    X = data[0,0]
    T = data[0,1]
    GPs = [GPCluster(rbf_kern,Y=X[list(cluster)],T=T[:,list(cluster)],var1=0.1, l=.05,var2=0.2) for cluster in clustering]
    likelihoods = [GPs[j].likelihood(X[i], T[:,i]) for j in range(len(GPs))]
    empty_gp = GPCluster(rbf_kern,Y=np.array([0]),T=np.array([0]),var1=.1, l=.01,var2=.1)
    likelihoods.append(empty_gp.likelihood_empty(X[i], T[:,i]))
    likelihoods = np.array(likelihoods)
    return likelihoods
# Settings tried:
#   1:same
#   1:same
def Py_z2(data, i, clustering, cluster_assn):
    Y = data[1,0]
    T = data[1,1]
    GPs = [GPCluster(rbf_kern,Y=Y[list(cluster)],T=T[:,list(cluster)],var1=.1, l=.05,var2=.2) for cluster in clustering]
    likelihoods = [GPs[j].likelihood(Y[i], T[:,i]) for j in range(len(GPs))]
    empty_gp = GPCluster(rbf_kern,Y=np.array([0]),T=np.array([0]),var1=.1, l=.05,var2=.1)
    likelihoods.append(empty_gp.likelihood_empty(Y[i], T[:,i]))
    return np.array(likelihoods)
