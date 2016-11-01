import numpy as np
import random
import scipy.stats as stats

from util import sample_multinomial, rbf_kern, load_simulation_data, \
                 multivariate_t_distribution, load_int_probe_data, load_ext_probe_data
from GPlikelihood import GPCluster
from DCRP import DCRP
import cPickle
import bokeh.plotting as bp
bp.output_notebook()

def compute_S(data):
    N,D = data.shape
    S = np.zeros((D,D))
    for n in range(N):
            S = S + np.dot(data[n].reshape(D,1), data[n].reshape(1,D))
    return S

# Load Data:
# Expected Data format:
#   data[0,0] = numpy array methylation
#   data[1,0] =
#   data[1,1] =

## TODO: You need to create a function similar to the one below to load data
data_ext = load_ext_probe_data()
N,_ = data_ext[0,0].shape

data_int = load_int_probe_data()
N,_ = data_int[0,0].shape

size_train = int(0.75 * N)
size_test = N - size_train

# Following is to avoid shallow copy of data:
# TODO: create a function to do this in utils
train_data_int = np.empty_like (data_int)
train_data_int[0,0] = np.empty_like (data_int[0,0][:size_train,:])
train_data_int[1,0] = np.empty_like (data_int[1,0][:size_train,:])
train_data_int[1,1] = np.empty_like (data_int[1,1][:,:size_train])
train_data_ext = np.empty_like (data_ext)
train_data_ext[0,0] = np.empty_like (data_ext[0,0][:size_train,:])
train_data_ext[1,0] = np.empty_like (data_ext[1,0][:size_train,:])
train_data_ext[1,1] = np.empty_like (data_ext[1,1][:,:size_train])

test_data_int = np.empty_like (data_int)
test_data_int[0,0] = np.empty_like (data_int[0,0][size_train:,:])
test_data_int[1,0] = np.empty_like (data_int[1,0][size_train:,:])
test_data_int[1,1] = np.empty_like (data_int[1,1][:,size_train:])
test_data_ext = np.empty_like (data_ext)
test_data_ext[0,0] = np.empty_like (data_ext[0,0][size_train:,:])
test_data_ext[1,0] = np.empty_like (data_ext[1,0][size_train:,:])
test_data_ext[1,1] = np.empty_like (data_ext[1,1][:,size_train:])



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
N,D = train_data_int[0,0].shape
prior1 = {
"m0" : np.mean(data_int[0,0], axis = 0).reshape(-1,1),
"k0": 10.,
"v0": 280.,
"s0": np.eye(D)
}
prior2 = {
"var1" : 0.01,
"var2": 0.05,
"length_scale": 0.5
}

rms_int = []
rms_ext = []
size_train = int(0.80 * N)
size_test = N - size_train
#TODO: start with a small number of experiments
num_experiments = 5
for iter in range(num_experiments):
    # idx is used to select random indices
    idx = np.random.choice(range(N),size=N,replace=False)

    train_data_int[0,0] = data_int[0,0][idx[:size_train],:]
    train_data_int[1,0] = data_int[1,0][idx[:size_train],:]
    train_data_int[1,1] = data_int[1,1][:,idx[:size_train]]
    # train_data_ext[0,0] = data_ext[0,0][idx[:size_train],:]
    # train_data_ext[1,0] = data_ext[1,0][idx[:size_train],:]
    # train_data_ext[1,1] = data_ext[1,1][:,idx[:size_train]]

    test_data_int[0,0] = data_int[0,0][idx[size_train:],:]
    test_data_int[1,0] = data_int[1,0][idx[size_train:],:]
    test_data_int[1,1] = data_int[1,1][:,idx[size_train:]]
    # test_data_ext[0,0] = data_ext[0,0][idx[size_train:],:]
    # test_data_ext[1,0] = data_ext[1,0][idx[size_train:],:]
    # test_data_ext[1,1] = data_ext[1,1][:,idx[size_train:]]
    dp1 = DCRP(alpha1,alpha2,Px_z1,Py_z2, prior1, prior2)
    # dp2 = DCRP(alpha1,alpha2,Px_z1,Py_z2, prior1, prior2)

    dp1._initialize_assn(train_data_int)
    # dp2._initialize_assn(train_data_ext)
    # Run gibbs sampling
    dp1._gibbs_sampling_crp(train_data_int)
    # dp2._gibbs_sampling_crp(train_data_ext)
    # Final clustering
    dp1_c2 = len(dp1.c2)
    # dp2_c2 = len(dp2.c2)
    dp1_gp = []
    # dp2_gp = []

    for k in range (dp1_c2):
        dp1_gp.append(GPCluster(rbf_kern,Y=train_data_int[1,0][list(dp1.c2[k]),:],\
                    T=train_data_int[1,1][:,list(dp1.c2[k])],var1=prior2["var1"], \
                    l=prior2["length_scale"],var2=prior2["var2"]))
    dp1_gp.append(GPCluster(rbf_kern,Y=np.array([[0,0,0,0]]),\
                    T=np.array([[0,1,2,3]]).T,var1=prior2["var1"],\
                    l=prior2["length_scale"],var2=prior2["var2"]))
    #
    # for k in range (dp2_c2):
    #     dp2_gp.append(GPCluster(rbf_kern,Y=train_data_ext[1,0][list(dp2.c2[k]),:],\
    #                 T=train_data_ext[1,1][:,list(dp2.c2[k])],var1=prior2["var1"], \
    #                 l=prior2["length_scale"],var2=prior2["var2"]))
    # dp2_gp.append(GPCluster(rbf_kern,Y=np.array([[0,0,0,0]]),\
    #                 T=np.array([[0,1,2,3]]).T,var1=prior2["var1"], \
    #                 l=prior2["length_scale"],var2=prior2["var2"]))

    ms = 0
    for m in range(size_test):
        y = test_data_int[1,0][m].reshape((-1,1))
        # y = test_data_ext[1,0][m].reshape((-1,1))
        y = y[:3]
        t = np.array([[0,1,2]])
        x = test_data_int[0,0][m]
        prob = dp1.predict(x,y,t[0:2],train_data_int)
        t = np.array([[0,1,2,3]])
        sum = 0
        for i in range(prob.shape[0]):
            sum = sum + prob[i]*dp1_gp[i].predict_y(y,t)[0]
        ms = ms + pow(sum - test_data_int[1,0][m,3],2)
    print "rms: ", np.sqrt(ms/size_test)
    rms_int.append(np.sqrt(ms/size_test))

print "rms: ", rms_int
