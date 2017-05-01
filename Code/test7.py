import numpy as np
import SpikeSimilarity
from time import time
import itertools as it
from sklearn.model_selection import train_test_split
import matplotlib as plt
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn import svm
from sklearn.cluster import dbscan
from sklearn.neighbors import KNeighborsClassifier
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

'''

for i in np.arange(1,6):
    start = time()
    data = np.load("../Data/All_{}_Contiguous.npy".format(i))
    print "Data {} Loaded in {}".format(i,time()-start)
    start = time()
    Xtr, Xte, Ytr, Yte = train_test_split(data[:,:-1],data[:,-1], test_size=0.2, train_size=0.8, random_state=0, stratify=data[:,-1])
    print "Data {} Split in {}".format(i,time()-start)
    start = time()
    np.save('../Data/Contiguous/sp{}_Ytr_contiguous.npy'.format(i),Ytr)
    np.save('../Data/Contiguous/sp{}_Yte_contiguous.npy'.format(i),Yte)
    print "Y {} Saved in {}".format(i,time()-start)
    start = time()
    d_m = pdist(Xtr, lambda u,v: SpikeSimilarity.spike_similarity(u,v))
    print "Computed pdist matrix {} in {}".format(i,time()-start)
    np.save('../Data/Contiguous/sp{}_Xtr_contiguous.npy'.format(i),d_m)
    np.save('../Data/Contiguous/sp{}_Xtr_nondm_contiguous.npy'.format(i),Xtr)
    start = time()
    t_m = cdist(Xte, Xtr, lambda u,v: SpikeSimilarity.spike_similarity(u,v))
    print "Computed cdist matrix {} in {}".format(i,time()-start)
    np.save('../Data/Contiguous/sp{}_Xte_contiguous.npy'.format(i),t_m)
    np.save('../Data/Contiguous/sp{}_Xte_nondm_contiguous.npy'.format(i),Xte)

# for i in np.arange(1,6):
#     data = np.load("../Data/All_{}_Contiguous.npy".format(i))
#     print data.shape


# labels = test_data[:,-1]
# test = test_data[:,:-1]
# print test_data.shape[0]
# print Xtr.shape
'''

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
# start = time()
# Xte = np.load("../Data/sp_xte.npy")
# Xtr = np.load("../Data/sp_xtr.npy")
# d_m = cdist( Xte,Xtr, lambda u,v: SpikeSimilarity.spike_similarity(u,v))
# np.save("../Data/sp_xtr_xte2.npy",d_m)
# print (time() - start)
#

# start = time()
# Xte = np.load("../Data/sp2_xte.npy")
# Xtr = np.load("../Data/sp2_xtr.npy")
# d_m = cdist(Xte, Xtr, lambda u,v: SpikeSimilarity.spike_similarity(u,v))
# np.save("../Data/sp2_xtr_xte.npy",d_m)
# print (time() - start)

for j in np.arange(1,6):
    Xtr = squareform(np.load("../Data/Contiguous/sp{}_Xtr_contiguous.npy".format(j)))

    print (Xtr.transpose() == Xtr).all()
    # Xtr = np.load("../Data/Contiguous/sp{}_Xtr_nondm_contiguous.npy".format(j))
    Ytr = np.load("../Data/Contiguous/sp{}_Ytr_contiguous.npy".format(j))
    # Xte = np.load("../Data/Contiguous/sp{}_Xte_nondm_contiguous.npy".format(j))
    Xte = np.load("../Data/Contiguous/sp{}_Xte_contiguous.npy".format(j))
    Yte = np.load("../Data/Contiguous/sp{}_Yte_contiguous.npy".format(j))

    # Xtr = squareform(np.load("../Data/sp2_distance_matrix.npy"))
    # Ytr = np.load("../Data/sp2_ytr.npy")
    # Xte = np.load("../Data/sp2_xtr_xte.npy")
    # Yte = np.load("../Data/sp2_yte.npy")


    # print Xte
    # print Yte

    # print Ytr[0:50]
    custom_dist = lambda u,v: SpikeSimilarity.spike_similarity(u,v)

    clf = svm.SVC(C=1.0, gamma=1.0,kernel='precomputed',decision_function_shape='ovr')
    clf.fit(Xtr,Ytr)
    # print clf.score(Xte,Yte)
    #
    # print clf.score(Xte,Yte)
    # print Xtr[0]
    # np.count_nonzero(Xte)
    #
    # print custom_dist(Xtr[0],Xtr[2])

    print np.hstack([Xtr[:,0].reshape(-1,1),Ytr.reshape(-1,1)])

    clf = KNeighborsClassifier(n_neighbors=5,metric='precomputed')
    clf.fit(Xtr,Ytr)
    # print clf.score(Xte,Yte)
''''''