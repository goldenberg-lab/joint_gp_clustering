import numpy as np
import random
from tqdm import tqdm
from util import sample_multinomial
import time

class DCRP(object):
    def __init__(self, alpha1,alpha2, Px_z1, Py_z2):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.Px_z1 = Px_z1
        self.Py_z2 = Py_z2
        #gibbs sampling parameters
        self.num_iter = 100
        self.eb_start = 20
        self.eb_interval = 5

    def cluster(self, data):
        return self._gibbs_sampling_crp(data)

    def _initialize_assn(self, d):
        """
        Initial cluster assignment before Gibbs sampling Process

        d[0,0] = X: N x t1 array
        d[0,1] = T1: t1 x N array
        d[1,0] = Y: N x t2 array
        d[1,1] = T2: t2 x N array
        """
        print "Initializing clustering"
        data = d[0,0]
        c1 = []
        ca1 = []
        c2 = []
        ca2 = []
        for i in range(len(data)):
            crp_prior1 = [(len(x) + 0.0) / (i + self.alpha1) for j, x in enumerate(c1)]
            crp_prior1.append(self.alpha1 / (i + self.alpha1))
            crp_prior1 = np.array(crp_prior1)
            l1 = self.Px_z1(d, i, c1, ca1)
            probs = crp_prior1 * l1
            cluster = sample_multinomial(probs)
            assert(cluster>=0)
            assert(cluster<=len(c1))
            if cluster == len(c1):
                s = set([i])
                c1.append(s)
            else:
                c1[cluster].add(i)
            ca1.append(c1[cluster])
        print "\tInitialized clusters for X"
        for i in range(len(data)):
            k = 0.
            crp_prior2 = []
            for j,x in enumerate(c2):
                k_ = len(ca1[i]&x)
                k = k + k_ + 1.
                crp_prior2.append(len(x) + k_)
            crp_prior2.append(self.alpha2)
            crp_prior2 = np.array(crp_prior2)/ (k+self.alpha2)
            l2 = self.Py_z2(d, i, c2, ca2)
            probs = crp_prior2 * l2

            cluster = sample_multinomial(probs)
            assert(cluster>=0)
            assert(cluster<=len(c2))
            if cluster == len(c2):
                s = set([i])
                c2.append(s)
            else:
                c2[cluster].add(i)
            ca2.append(c2[cluster])
        print "\tInitialized clusters for Y"
        print "Done Initialization"
        self.c1, self.ca1,self.c2,self.ca2 = (c1, ca1, c2, ca2)
        return 

    def _gibbs_sampling_crp(self, data):
        """
        Run Gibbs sampling to get the cluster assignment
        
        data[0,0] = X: N x t1 array
        data[0,1] = T1: t1 x N array
        data[1,0] = Y: N x t2 array
        data[1,1] = T2: t2 x N array
        """
        print "Running Gibbs Sampler"
        num_data = len(data)
        c1, ca1,c2,ca2 = (self.c1, self.ca1,self.c2,self.ca2)
        for t in tqdm(range(self.num_iter)):
            num_new_clusters1 = 0.0
            num_new_clusters2 = 0.0
            for i in range(num_data):
                ## Part 1
                ca1[i].remove(i)
                if len(ca1[i]) == 0:
                    c1.remove(ca1[i])
                cp1 = [(len(x) + 0.0) / (num_data - 1 + self.alpha1) for j, x in enumerate(c1)]
                cp1.append(self.alpha1 / (num_data - 1 + self.alpha1))
                cp1 = np.array(cp1)
                l1 = self.Px_z1(data, i, c1, ca1)# likelihood for each cluster
                p1 = cp1 * l1

                cluster = sample_multinomial(p1)
                if cluster == len(c1):
                    s = set([i])
                    c1.append(s)
                    num_new_clusters1 += 1
                else:
                    c1[cluster].add(i)
                ca1[i] = c1[cluster]
                ## Part 2
                ca2[i].remove(i)
                if len(ca2[i]) == 0:
                    c2.remove(ca2[i])
                cp2 = [(len(x) + 0.0) for j, x in enumerate(c2)]
                cp2.append(self.alpha2)
                cp2 = np.array(cp2)
                cp2 = cp2 / np.sum(cp2)
                l2 = self.Py_z2(data, i, c2, ca2)
                p2 = cp2 * l2

                cluster = sample_multinomial(p2)
                if cluster == len(c2):
                    s = set([i])
                    c2.append(s)
                    num_new_clusters2 += 1
                else:
                    c2[cluster].add(i)
                ca2[i] = c2[cluster]

            # Empirical Bayes for adjusting hyperparameters
            if t % self.eb_interval == 0 and t > self.eb_start:
                self.alpha1 = num_new_clusters1
                self.alpha2 = num_new_clusters2

        self.c1, self.ca1,self.c2,self.ca2 = (c1, ca1,c2,ca2)
        return
