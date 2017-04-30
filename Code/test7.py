import numpy as np
import SpikeSimilarity
from time import time
import itertools as it
from sklearn.model_selection import train_test_split
import matplotlib as plt
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster import dbscan
# data = np.load("../Data/All_1.npy")
# print data[0,np.nonzero(data[0,:-1])]
# print data[1,np.nonzero(data[1,:-1])]
# print data[0,-1]
# print data[1,-1]
# print np.corrcoef(data[0,:-1],data[1,:-1])

# cond = (data[:,-1] == 0)
# label_zero = data[cond]
# cond = (data[:,-1] == 1)
# label_one = data[cond]
# cond = (data[:,-1] == 2)
# label_two = data[cond]
# cond = (data[:,-1] == 3)
# label_three = data[cond]
# np.random.seed(0)
# test_data = np.vstack([label_zero[0:60],label_one[200:260],label_three[1000:1060],label_two[2000:2060]])
# np.random.shuffle(test_data)
# Xtr, Xte, Ytr, Yte = train_test_split(data[:,:-1],data[:,-1], test_size=0.1, train_size=0.20, random_state=67, stratify=data[:,-1])
# labels = test_data[:,-1]
# test = test_data[:,:-1]
# print test_data.shape[0]
# print Xtr.shape


'''
TESTING PYSPIKE

spike_trains = spk.load_spike_trains_from_txt("./pyspike/PySpike_testdata.txt",edges=(0,4000))
print spike_trains[0]
# isi_profile = spk.isi_profile()



TESTING PYSPIKE
'''
# weights = {"add":1.0, "delete":1.0, "move":1.0/200}
# start = time()
# # print SpikeSimilarity.spike_similarity(test[0],test[1])
# d_m = pdist(Xtr, lambda u,v: SpikeSimilarity.spike_similarity(u,v))
# # d_m = [SpikeSimilarity.spike_similarity(x,y) for x,y in it.izip(test[1:], test)]
# # print SpikeSimilarity.spike_similarity(test[0],test[9])
# # print test_data[2]
# print (time() - start)
# # print d_m.shape
# # start = time()
# np.save("../Data/sp2_xte.npy",Xte)
# np.save("../Data/sp2_yte.npy",Yte)
# np.save("../Data/sp2_ytr.npy",Ytr)
# np.save("../Data/sp2_xtr.npy",Xtr)
# np.save("../Data/sp2_distance_matrix.npy",d_m)
# print (time() - start)
start = time()
Xte = np.load("../Data/sp_xte.npy")
Xtr = np.load("../Data/sp_xtr.npy")
d_m = cdist(Xtr, Xte, lambda u,v: SpikeSimilarity.spike_similarity(u,v))
np.save("../Data/sp_xte_xte",d_m)
print (time() - start)


start = time()
Xte = np.load("../Data/sp2_xte.npy")
Xtr = np.load("../Data/sp2_xtr.npy")
d_m = cdist(Xtr, Xte, lambda u,v: SpikeSimilarity.spike_similarity(u,v))
np.save("../Data/sp2_xte_xte",d_m)
print (time() - start)

