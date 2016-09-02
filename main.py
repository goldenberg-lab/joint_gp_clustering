import numpy as np
import random
import scipy.stats as stats

from util import sample_multinomial, rbf_kern, load_simulation_data
from GPlikelihood import GPCluster
from DCRP import DCRP
import cPickle

def Px_z1(data, i, clustering, cluster_assn):
    # NOTE: Set the constants according to data
    # data[0,0]: X - N x k numpy array
    X = data[0,0]
    n,k = X.shape
    var1 = 1
    var2 = 2
    means = [np.mean(X[list(cluster)],axis=0) for cluster in clustering]
    means.append(np.zeros(k))
    stds = [np.std(X[list(cluster)],axis=0) + var1 for cluster in clustering]
    std = np.zeros(k)+ var2
    stds.append(std)
    return np.array([stats.multivariate_normal.pdf(data[i], means[j], stds[j]) for j in range(len(means))])


def Py_z2(data, i, clustering, cluster_assn):
    Y = data[1,0]
    T = data[1,1]

    ## NOTE: Should change var1, l and var2
    GPs = [GPCluster(rbf_kern,Y=Y[list(cluster)],T=T[:,list(cluster)],var1=.1, l=.05,var2=.2) for cluster in clustering]
    likelihoods = [GPs[j].likelihood(Y[i], T[:,i]) for j in range(len(GPs))]
    empty_gp = GPCluster(rbf_kern,Y=np.array([0]),T=np.array([0]),var1=.1, l=.05,var2=.1)
    likelihoods.append(empty_gp.likelihood_empty(Y[i], T[:,i]))
    return np.array(likelihoods)


data = load_simulation_data()
alpha1 = 1.
alpha2 = 1.
dp = DCRP(alpha1,alpha2,Px_z1,Py_z2)
dp._initialize_assn(data)
print "Initial number of clusters c1: ", len(dp.c1)
print "Initial number of clusters c2: ", len(dp.c2)
dp._gibbs_sampling_crp(data)
print "Final number of clusters c1: ", len(dp.c1)
print "Final number of clusters c2: ", len(dp.c2)
cPickle.dump(dp,open("dp2.dat","wb"))
cPickle.dump(data,open("data2.dat","wb"))
