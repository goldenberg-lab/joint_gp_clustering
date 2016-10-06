import numpy as np
import random
import scipy.stats as stats

from util import sample_multinomial, rbf_kern, load_simulation_data, multivariate_t_distribution
from GPlikelihood import GPCluster
from DCRP import DCRP
import cPickle


def Px_z1(data, i, clustering, cluster_assn, prior):
    # NOTE: Set the constants according to data
    # data[0,0]: X - N x k numpy array
    X = data[0,0]
    n,d = X.shape

    if i == int:
        new = X[i].reshape(1,d)
    else:
        new = i
    m0 =  prior["m0"]
    k0 = prior["k0"]
    v0 = prior["v0"]
    S0 = prior["s0"]
    ret = []
    for cluster in clustering:
        data_k = X[list(cluster)]
        Nk = X[list(cluster)].shape[0]
        k_Nk = k0 + Nk
        v_Nk = v0 + Nk
        means = np.mean(data_k,axis=0).reshape(d,1)
        assert(means.shape == (d,1))
        m_Nk = (k0 * m0 + Nk * means) / k_Nk
        S = np.zeros((d,d))

        for n in range(Nk):
            S = S + np.dot(data_k[n].reshape(d,1), data_k[n].reshape(d,1).T)
        ti = k_Nk * np.dot(m_Nk.reshape(d,1),m_Nk.reshape(d,1).T)
        S_Nk_d = (k_Nk+1.)/(k_Nk*(v_Nk - d + 1))*(S0 + S + k0*np.dot(m0.reshape(d,1), m0.reshape(d,1).T) - k_Nk * np.dot(m_Nk.reshape(d,1),m_Nk.reshape(d,1).T))

        assert np.linalg.det(S_Nk_d) > 0
        ret.append(multivariate_t_distribution(X[i].reshape(d,1), m_Nk, S_Nk_d, v0-d+1, d))
    S_d = (k0 + 1.)/(k0*(v0-d+1)) * (S0)
    ret.append(multivariate_t_distribution(X[i].reshape(d,1), m0, S_d,v0-d+1, d))

    return np.array(ret)


def Py_z2(data, i, clustering, cluster_assn, prior, ti=0):
    Y = data[1,0]
    T = data[1,1]
    if type(i) == int:
        new_y = Y[i]
        new_t = T[:,i]
    else:
        new_y = i
        new_t = ti
    ## NOTE: Should change var1, l and var2
    v1 = prior["var1"] # 0.1
    v2 = prior["var2"] # 0.2
    l = prior["length_scale"] # 0.05
    GPs = [GPCluster(rbf_kern,Y=Y[list(cluster)],T=T[:,list(cluster)],var1=v1, l=l,var2=v2) for cluster in clustering]
    likelihoods = [GPs[j].likelihood(new_y, new_t) for j in range(len(GPs))]
    empty_gp = GPCluster(rbf_kern,Y=np.array([0]),T=np.array([0]),var1=v1, l=l,var2=v2)
    likelihoods.append(empty_gp.likelihood_empty(new_y, new_t))
    return np.array(likelihoods)

data = load_simulation_data()

int_data_train, int_data_test = load_int_data()
ext_data_train, ext_data_test = load_ext_data()

alpha1 = 1.
alpha2 = 1.
prior1 = {
"m0" : 0.,
"k0": 1.,
"v0": 1.,
"s0_scale": 1.
}
prior2 = {
"var1" : 0.,
"var1": 1.,
"length_scale": 1.
}
dp = DCRP(alpha1,alpha2,Px_z1,Py_z2, prior1, prior2)

dp._initialize_assn(data)
print "Initial number of clusters c1: ", len(dp.c1)
print "Initial number of clusters c2: ", len(dp.c2)
dp._gibbs_sampling_crp(data)
print "Final number of clusters c1: ", len(dp.c1)
print "Final number of clusters c2: ", len(dp.c2)
cPickle.dump(dp,open("dp2.dat","wb"))
cPickle.dump(data,open("data2.dat","wb"))

# idea is to get mean and variance of a particular point
