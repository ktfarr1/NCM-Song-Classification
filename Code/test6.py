import numpy as np
import glob
import DataProcessing
from time import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

if __name__ == '__main__':
# for b in [2,4,8,16,32,64,128,256]:
# # for b in [1]:
#     for p in [150820,155665,160510]:
#     # for p in [155665]:
#         key_paths = np.array(glob.glob("../Data/{}/Keys/*.csv".format(p)))
#         mem_paths = np.array(glob.glob("../Data/{}/Memory/*.txt".format(p)))
#         labels = np.asarray([path.lstrip("..\\Data\\{}\\Keys\\".format(p)).upper().split('_')[0:2] for path in key_paths])
#         data = DataProcessing.stack_arrays(p,mem_paths,key_paths,labels,b)
#
#         np.save("../Data/{}_{}.npy".format(p,b),data)

# for b in [2,4,8,16,32,64,128,256]:
#     list = glob.glob("../Data/*_{}.npy".format(b))
#     print list
#     full = np.vstack([np.load(l) for l in list])
#     np.save("../Data/All_{}.npy".format(b),full)
#     # print np.sum(full)

    for b in [2,4,8,16,32,64,128,256]:
        start = time()
        data = np.load("../Data/All_{}.npy".format(b))
        print "Loaded data {} in {} sec".format(b,time()-start)
        start = time()
        Xtr, Xte, Ytr, Yte = train_test_split(data[:,:-1],data[:,-1], test_size=0.4, train_size=0.6, random_state=0, stratify=data[:,-1])
        print "Split data {} in {} sec".format(b, time() - start)
        C_s = np.logspace(-3, 3, 25)
        G_s = np.logspace(-5,2,25,base=(data.shape[1]-1))
        params = {"C": C_s, "gamma": G_s}
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
        # print clf.get_params().keys()
        rand_search = RandomizedSearchCV(clf,params,n_jobs=2,refit=True,random_state=0,n_iter=10)
        start = time()
        rand_search.fit(Xtr, Ytr)
        print "Random Search took {} for 20 candidates".format(time()-start)
        print "Bin Size: {}".format(b)
        print rand_search.best_params_
        # print rand_search.cv_results_
        print rand_search.best_estimator_.score(Xte,Yte)