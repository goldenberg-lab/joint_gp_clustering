import numpy as np
import random
import scipy.stats as stats
from GPlikelihood import GPCluster
import time
import csv
from math import *

def load_int_data_mrmr_10probes_cnsdev():
    probes = []
    with open('./data/mrmr-selected-10-cns-dev-probes-10-23-2016.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            probes.append(np.array(row))
    probes = np.array(probes)
 #   probes = probes[:,:].astype("double") # removed [no_missing_indx,:]
    internalizing = []
    with open('./data/internalizing-nomissing-zvals-ts-10-23-2016.txt') as csvfile:
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

def load_ext_data_mrmr_10probes_cnsdev():
    probes = []
    with open('./data/mrmr-selected-10-cns-dev-probes-10-23-2016.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            probes.append(np.array(row))
    probes = np.array(probes[1:])[:,1:].T
    probes = probes.astype("double")  # removed [no_missing_indx,:]
    externalizing = []
    with open('./data/externalizing-nomissing-zvals-ts-10-23-2016.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            externalizing.append(np.array(row))
    externalizing = np.array(externalizing)
    externalizing_id = externalizing[1]
    externalizing_z = np.array(externalizing[1:,5:]).astype("double")
    externalizing_raw = np.array(externalizing[1:,1:5]).astype("double")
    T = np.zeros(externalizing_z.shape)
    T[:,0] = 0
    T[:,1] = 6
    T[:,2] = 30
    T[:,3] = 42
    return np.array([[probes,[0]],[externalizing_z,T.T]])

def load_ext_data_mrmr_10probes_MAVAN_candidate():
    probes = []
    with open('../../../mrmr-selected-10-mavan-cand-probes-10-23-2016.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            probes.append(np.array(row))
    probes = np.array(probes[1:])[:,1:].T
    probes = probes.astype("double")  # removed [no_missing_indx,:]
    externalizing = []
    with open('../../../externalizing-nomissing-zvals-ts-10-23-2016.txt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            externalizing.append(np.array(row))
    externalizing = np.array(externalizing)
    externalizing_id = externalizing[1]
    externalizing_z = np.array(externalizing[1:,5:]).astype("double")
    externalizing_raw = np.array(externalizing[1:,1:5]).astype("double")
    T = np.zeros(externalizing_z.shape)
    T[:,0] = 0
    T[:,1] = 6
    T[:,2] = 30
    T[:,3] = 42
    return np.array([[probes,[0]],[externalizing_z,T.T]])
