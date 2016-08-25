import numpy as np
import bokeh.plotting as bp
bp.output_notebook()
import random
import scipy.stats as stats

from util import sample_multinomial, rbf_kern, Px_z1, Py_z2, load_simulation_data
from GPlikelihood import GPCluster
from DCRP import DCRP
import cPickle

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


