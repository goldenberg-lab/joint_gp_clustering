import numpy as np
import random
import scipy.stats as stats

from util import sample_multinomial, rbf_kern, load_simulation_data, \
                 multivariate_t_distribution, load_int_probe_data, load_ext_probe_data
from GPlikelihood import GPCluster
from DCRP import DCRP
import cPickle

def compute_S(data):
    N,D = data.shape
    S = np.zeros((D,D))
    for n in range(N):
            S = S + np.dot(data[n].reshape(D,1), data[n].reshape(1,D))
    return S

# Load Data:
data = load_simulation_data()
N= data[0,0].shape[0]
train_data = np.empty_like (data)
size_train = int(0.80 * N)
size_test = N - size_train
train_data[0,0] = np.empty_like (data[0,0][:size_train,:])
train_data[1,0] = np.empty_like (data[1,0][:size_train,:])
train_data[1,1] = np.empty_like (data[1,1][:,:size_train])

test_data = np.empty_like (data)
test_data[0,0] = np.empty_like (data[0,0][size_train:,:])
test_data[1,0] = np.empty_like (data[1,0][size_train:,:])
test_data[1,1] = np.empty_like (data[1,1][:,size_train:])

# Gaussian Likelihood:
def Px_z1(data, i, clustering, cluster_assn, prior):
    # NOTE: Set the constants according to data
    # data[0,0]: X - N x k numpy array
    X = data[0,0]
    N,D = X.shape

    if type(i) == int:
        new_x = X[i]
    else:
        new_x = np.array(i)

    m0 =  prior["m0"].reshape(D,1)
    k0 = prior["k0"]
    v0 = prior["v0"]
    S0 = prior["s0"]
    ret = []
    for cluster in clustering:
        data_k = X[list(cluster)]
        Nk = X[list(cluster)].shape[0] + 0.0
        kNk = k0 + Nk
        vNk = v0 + Nk
        mNk = 1. / kNk * (k0 * m0 + Nk* np.mean(data_k,axis=0).reshape(D,1))
        S = compute_S(data_k)
        SNk = S0 + S + k0*np.dot(m0.reshape(D,1), m0.reshape(D,1).T) - \
                kNk*np.dot(mNk.reshape(D,1), mNk.reshape(D,1).T)
        prop = (kNk + 1.)/(kNk*(vNk-D+1.))
        ret.append(multivariate_t_distribution(new_x.reshape(D,1), mNk, prop*SNk, vNk-D+1, D))
    prop = (k0 + 1.) /(k0*(v0-D+1))
    ret.append(multivariate_t_distribution(new_x.reshape(D,1), m0, prop*S0 ,v0-D+1, D))

    return np.array(ret)

# GP Likelihood
def Py_z2(data, i, clustering, cluster_assn, prior, ti=0):
    Y = data[1,0]
    T = data[1,1]
    if type(i) == int:
        new_y = Y[i]
        new_t = T[:,i]
    else:
        new_y = i
        new_t = ti
    v1 = prior["var1"]
    v2 = prior["var2"]
    l = prior["length_scale"]
    GPs = [GPCluster(rbf_kern,Y=Y[list(cluster)],T=T[:,list(cluster)],var1=v1, l=l,var2=v2)\
                                                                 for cluster in clustering]
    likelihoods = [GPs[j].likelihood(new_y, new_t) for j in range(len(GPs))]
    empty_gp = GPCluster(rbf_kern,Y=np.array([0]),T=np.array([0]),var1=v1, l=l,var2=v2)
    likelihoods.append(empty_gp.likelihood_empty(new_y, new_t))
    return np.array(likelihoods)


# Hyperparameters
# TODO: Need to try different hyperparameters
alpha1 = 1.
alpha2 = 1.
N,D = train_data[0,0].shape
prior1 = {
"m0" : np.array([[0,0,0]]),
"k0": 0.1,
"s0": 3*np.eye(D),
"v0": 4.
}
prior2 = {
"var1" : 0.06, # variance for being
"var2": 0.15,
"length_scale": .6
}
# print prior1
print prior2
rms_int = []
rms_ext = []

#TODO: start with a small number of experiments

num_experiments = 15
for iter in range(num_experiments):
    # idx is used to select random indices
    num_total= data[0,0].shape[0]
    idx = np.random.choice(range(num_total),size=num_total,replace=False)

    train_data[0,0] = data[0,0][idx[:size_train],:]
    train_data[1,0] = data[1,0][idx[:size_train],:]
    train_data[1,1] = data[1,1][:,idx[:size_train]]


    test_data[0,0] = data[0,0][idx[size_train:],:]
    test_data[1,0] = data[1,0][idx[size_train:],:]
    test_data[1,1] = data[1,1][:,idx[size_train:]]

    dp = DCRP(alpha1,alpha2,Px_z1,Py_z2, prior1, prior2)

    dp._initialize_assn(train_data)
    print "Number of clusters for X = ", len(dp.c1)
    print "Number of clusters for Y = ", len(dp.c2)
    # Run gibbs sampling
    dp._gibbs_sampling_crp(train_data)
    print "Number of clusters for X = ", len(dp.c1)
    print "Number of clusters for Y = ", len(dp.c2)
    # Final clustering
    dp_c2 = len(dp.c2)
    dp_gp = []

    for k in range (dp_c2):
        dp_gp.append(GPCluster(rbf_kern,Y=train_data[1,0][list(dp.c2[k]),:],\
                    T=train_data[1,1][:,list(dp.c2[k])],var1=prior2["var1"], \
                    l=prior2["length_scale"],var2=prior2["var2"]))
    dp_gp.append(GPCluster(rbf_kern,Y=np.array([[0,0,0,0,0]]),\
                    T=np.array([[0,1,2,3,4]]).T,var1=prior2["var1"],\
                    l=prior2["length_scale"],var2=prior2["var2"]))

    ms = 0
    for m in range(size_test):
        y = test_data[1,0][m].reshape((-1,1))
        y = y[:4]
        t = np.array([[0,1,2,3]])
        x = test_data[0,0][m]
        prob = dp.predict(x,y,t,train_data)
        t = np.array([[0,1,2,3,4]])
        sum = 0
        for i in range(prob.shape[0]):
            sum = sum + prob[i]*dp_gp[i].predict_y(y,t)[0]
        print "Predicted value = ", sum
        print "Actual value = ", test_data[1,0][m,4]
        ms = ms + pow(sum - test_data[1,0][m,4],2)
    print "rms: ", np.sqrt(ms/size_test)
# rms_int.append(np.sqrt(ms/size_test))
