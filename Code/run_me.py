import numpy as np
import glob
import DataProcessing
from time import time
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.spatial.distance import squareform
import SpikeSimilarity
import Figures
from sklearn import preprocessing
from sklearn.manifold import TSNE,MDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD, PCA

if __name__ == '__main__':
    '''
    Uncomment to create data sets from raw data
    '''
    '''
    for b in [1,2,4,8,16,40,80,125,250]:
        start = time()
        for p in [150820,155665,160510]:
            key_paths = np.array(glob.glob("../Data/{}/Keys/*.csv".format(p)))
            mem_paths = np.array(glob.glob("../Data/{}/Memory/*.txt".format(p)))
            labels = np.asarray([path.lstrip("..\\Data\\{}\\Keys\\".format(p)).upper().split('_')[0:2] for path in key_paths])
            data = DataProcessing.stack_arrays(p,mem_paths,key_paths,labels,b)
            np.save("../Data/{}_{}.npy".format(p,b),data)
        print "Finished bin size {}ms in {}sec".format(b,time()-start)

    for b in [1,2,4,8,16,40,80,125,250]:
        list = glob.glob("../Data/*_{}.npy".format(b))
        print list
        full = np.vstack([np.load(l) for l in list])
        np.save("../Data/All_{}.npy".format(b),full)
    '''




    '''
    Basic SVC Operation on data set, using RBF kernel
    Hyperparameter optimization is done using Randomized Search
    '''

    ''''''
    best_params = {}
    best_score = {}
    # best_estimator = {}
    boost_score = {}
    boost_probs = {}

    # for b in [1,2,4,8,16,40,80,125,250]:
    for b in [8,16,40,80,125]:
        data = np.load("../Data/All_{}.npy".format(b))
        start = time()
        Xtr, Xte, Ytr, Yte = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, train_size=0.8, random_state=0,
                                              stratify=data[:, -1])
        print "Bin {} Split 1 in {}".format(b,time()-start)
        start = time()
        xtr, xte, ytr, yte = train_test_split(Xtr, Ytr, test_size=0.2, train_size=0.8, random_state=0,
                                          stratify=Ytr)
        print "Bin {} Split 2 in {}".format(b,time()-start)
        C_s = np.logspace(-3, 3, 25)
        G_s = np.logspace(-5,2,25,base=(data.shape[1]-1))
        params = {"C": C_s, "gamma": G_s}
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
        # print clf.get_params().keys()
        rand_search = RandomizedSearchCV(clf,params,n_jobs=2,refit=True,random_state=0,n_iter=15)
        start = time()
        rand_search.fit(xtr, ytr)
        print "Random Search took {} for 15 candidates".format(time()-start)
        best_params["{}".format(b)] = rand_search.best_params_
        # best_estimator["{}".format(b)] = rand_search.best_estimator_
        best_score["{}".format(b)] = rand_search.best_estimator_.score(xte,yte)

        # Use adaboost on full training set, test on withheld portion

        base_clf = svm.SVC(kernel='rbf', probability=True, decision_function_shape='ovr')
        base_clf.set_params(C=best_params[str(b)]['C'], gamma=best_params[str(b)]['gamma'])
        ensemble = AdaBoostClassifier(base_estimator=base_clf, random_state=0,n_estimators=50)
        # base_clf.fit(Xtr,Ytr)
        start = time()
        ensemble.fit(Xtr, Ytr)
        score = ensemble.score(Xte, Yte)
        # score = base_clf.score(Xte,Yte)
        boost_score[b] = score
        probs = ensemble.predict_proba(Xte)
        # probs = base_clf.predict_proba(Xte)
        boost_probs[b] = probs
        preds = ensemble.predict(Xte)
        # preds = base_clf.predict(Xte)
        print "Binsize: {} Score: {} Time: {}".format(b, score, time() - start)
        Figures.plot_confusion_matrix(Yte,preds, True,"AdaBoost {}ms Bins".format(b))
        # Figures.plot_confusion_matrix(Yte,preds,True,"SVM {} Bins".format(b))




    '''
    Testing SVM with Custom Distance function (provided by spike similarity)
    Note: This is using pre-computed distance matrices, as the function I wrote to
    score spike trains is completely unoptimized, and extremely slow.
    
    As such, I pre-computed 5 distance matrices on train-test splits of overall data.
    Given the amount of time and memory, these slices are training on ~15% of the total,
    and testing on ~5%
    '''


    '''

    for c in np.arange(1,6):
        # sca = preprocessing.Normalizer(norm='l2')
        sca = preprocessing.StandardScaler(copy=True,with_mean=True,with_std=False)
        Xtr = squareform(np.load("../Data/Numpy/Custom Distance/sp{}_Xtr.npy".format(c)))
        Xte = np.load("../Data/Numpy/Custom Distance/sp{}_Xte.npy".format(c))
        # Xtr = sca.fit_transform(np.load("../Data/Numpy/Custom Distance/sp{}_Xtr_nondm.npy".format(c)))
        # Xtr = np.load("../Data/Numpy/Custom Distance/sp{}_Xtr_nondm.npy".format(c))
        # Xte = sca.fit_transform(np.load("../Data/Numpy/Custom Distance/sp{}_Xte_nondm.npy".format(c)))
        # Xte = np.load("../Data/Numpy/Custom Distance/sp{}_Xte_nondm.npy".format(c))
        # print "Min,mean,max of Xtr"
        # print np.min(Xtr)
        # print np.mean(Xtr)
        # print np.max(Xtr)
        # print "Min, mean, max of Xte"
        # print np.min(Xte)
        # print np.mean(Xte)
        # print np.max(Xte)
        print Xtr

        Ytr = np.load("../Data/Numpy/Custom Distance/sp{}_Ytr.npy".format(c))
        Yte = np.load("../Data/Numpy/Custom Distance/sp{}_Yte.npy".format(c))

        # clf = svm.SVC(C=1.0,kernel='precomputed',decision_function_shape="ovr",probability=True)
        clf = neighbors.RadiusNeighborsClassifier(radius=25.0, metric='precomputed',p=2,outlier_label=-1)
        clf.fit(Xtr,Ytr)
        print "Score:"
        # print clf.score(Xte,Yte)
        print Yte.shape
        print len(clf.predict(Xte) == -1)
    '''

    '''
    Truncated SVD Linear Dimensionality Reduction into t-distributed Stochastic Neighbor Embedding
    This section loads the random single units (with normal Xtr and Xte arrays, as well as the corresponding
    precomputed distance matrices, so the classification can be switched between precomputed and a different
    supported kernel metric.
    
    '''

    '''
    # Uncomment this code to create arrays of multiple random cells, which are then used to test the custom similarity function

    # for b in [1,2,3,4,5]:
    #     start = time()
    #     for p in [150820, 155665, 160510]:
    #         np.random.seed(b)
    #         key_paths = np.array(glob.glob("../Data/{}/Keys/*.csv".format(p)))
    #         mem_paths = np.array(glob.glob("../Data/{}/Memory/*.txt".format(p)))
    #         choice = np.random.choice(key_paths.shape[0],size=4,replace=False)
    #         key_paths = key_paths[choice]
    #         mem_paths = mem_paths[choice]
    #         labels = np.asarray(
    #             [path.lstrip("..\\Data\\{}\\Keys\\".format(p)).upper().split('_')[0:2] for path in key_paths])
    #         data = DataProcessing.stack_arrays(p, mem_paths, key_paths, labels, 1)
    #         np.save("../Data/{}_{}_Contiguous.npy".format(p, b), data)
    #     list = glob.glob("../Data/*_{}_Contiguous.npy".format(b))
    #     print list
    #     full = np.vstack([np.load(l) for l in list])
    #     np.save("../Data/All_{}_Contiguous.npy".format(b),full)
    #     print "Finished bin size {}ms in {}sec".format(b, time() - start)

    # for b in [1,2,3,4,5]:
    #     start = time()

    '''

    '''
    for j in np.arange(1, 6):
        Xtr_d = squareform(np.load("../Data/Contiguous/sp{}_Xtr_contiguous.npy".format(j)))
        Xtr_n = Xtr_d / np.sum(Xtr_d)
        Xtr = np.load("../Data/Contiguous/sp{}_Xtr_nondm_contiguous.npy".format(j))
        Ytr = np.load("../Data/Contiguous/sp{}_Ytr_contiguous.npy".format(j))
        Xte = np.load("../Data/Contiguous/sp{}_Xte_nondm_contiguous.npy".format(j))
        Xte = np.load("../Data/Contiguous/sp{}_Xte_contiguous.npy".format(j))
        Xte_n = Xte / np.sum(Xte)
        Yte = np.load("../Data/Contiguous/sp{}_Yte_contiguous.npy".format(j))


        clf = svm.SVC(C=1.0, kernel='precomputed', decision_function_shape="ovr", probability=True)
        clf.fit(Xtr_n,Ytr)
        print clf.predict_proba(Xte_n)
        trunc = TruncatedSVD(n_components=50, algorithm='randomized', n_iter=5, random_state=0, tol=0.0)
        pca = PCA(n_components=50,copy=True,random_state=0)
        # t_data = trunc.fit_transform(Xtr)
        # p_data = pca.fit_transform(Xtr)
        # print t_data[1]
        # print np.count_nonzero(t_data)
        #TSNE Perplexity 5-50, LR 100-1000,

        cluster = TSNE(n_components=3, perplexity=5.0,early_exaggeration=4.0,learning_rate=1000.0,n_iter=1000,metric='precomputed')
        # cluster = TSNE(n_components=3, perplexity=5.0,early_exaggeration=4.0,learning_rate=1000.0,n_iter=1000,metric='euclidean')
        # c = cluster.fit_transform(t_data)
        c = cluster.fit_transform(Xtr_d)
        # plt.scatter(c[:,0],c[:,1],c=Ytr)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(c[:,0],c[:,1],c[:,2],c=Ytr)
        plt.show()
    '''

    '''
    Single Cell t-SNE visualization
    '''
    # mem_paths = ['../Data/150820/Memory/EMEK022_unitA_Memory.txt','../Data/150820/Memory/EMEK048_unitB_Memory.txt','../Data/155665/Memory/GPAK009_UNITB_MEMORY_1.txt','../Data/155665/Memory/GPAK005_UNITA_MEMORY_1.txt','../Data/160510/Memory/GPAK032_UNITB_MEMORY_1.txt']
    # key_paths = ['../Data/150820/Keys/EMEK022_unitA_key.csv','../Data/150820/Keys/EMEK048_unitB_key.csv','../Data/155665/Keys/GPAK009_UNITB_KEY_1.csv','../Data/155665/Keys/GPAK005_UNITA_KEY_1.csv','../Data/160510/Keys/GPAK032_UNITB_KEY_1.csv']
    #
    # for b in [1,8,16,40,80,125,250]:
    #     for i in range(5):
    #         np.save("../Data/Numpy/Single Unit/Single Unit {} {}".format(i,b),DataProcessing.array_from_raw_data(0,mem_paths[i],key_paths[i],[],b))

    for j in range(5):

        X = squareform(np.load('../Data/Numpy/Single Unit/Single Unit DM {} 1.npy'.format(j)))
        X_n = X / np.sum(X)
        data = np.load('../Data/Numpy/Single Unit/Single Unit {} 1.npy'.format(j))
        X = data[:,:-1]
        Y = data[:,-1]

        # TSNE Perplexity 5-50, LR 100-1000,

        cluster = TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, metric='precomputed')
        c = cluster.fit_transform(X_n)
        p = plt.scatter(c[:,0],c[:,1],c=Y)
        plt.savefig("../Figures/Single Unit {} 2d.png".format(j))

        # cluster = TSNE(n_components=3, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, metric='precomputed')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=Y)
        # plt.savefig("../Figures/Single Unit {} 3d.png".format(j))


    # for k in range(5):
    #     data = np.load('../Data/Numpy/Single Unit/Single Unit {} 16.npy'.format(k))
    #     X = data[:,:-1]
    #     X_n = X/np.sum(X)
    #     Y = data[:,-1]
    #     cluster = TSNE(n_components=3, perplexity=50.0, early_exaggeration=4.0, learning_rate=250.0, n_iter=1000,
    #                    metric='manhattan')
    #     c = cluster.fit_transform(X_n)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=Y)
    #     plt.show()