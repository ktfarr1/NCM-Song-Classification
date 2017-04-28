import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import decomposition
from fractions import Fraction
import glob
from sklearn import svm
import time
import copy
if __name__ == '__main__':
	npy_paths = glob.glob("..\\Data\\Numpy\\150820\\*.npy")

	data = np.empty((len(npy_paths)*80,126))
	for i in range(len(npy_paths)):
		data[(80*i):((i+1)*80)] = np.load(npy_paths[i])
	data_points = data.shape[0]
	n_features = data.shape[1]-1
	np.random.seed(0)
	np.random.shuffle(data)
	labels = np.asarray(data[:,-1]).reshape(-1,1)
	samples = data[:,:-1]

	normed = preprocessing.Normalizer(norm='l2',copy=True).fit_transform(samples)
	for i in np.arange(2,5):
		pca = decomposition.PCA(n_components=20*i,copy=True, svd_solver='full').fit_transform(normed)
		print pca.shape, labels.shape
		pca_data = np.hstack([pca, labels])
		# pca_data = np.hstack([normed, labels.reshape(-1,1)])
		C_s = np.logspace(-5,5,10)
		G_s = np.logspace(-5,1,10,base=(pca_data.shape[0]-1))

		parameters = np.transpose([np.tile(C_s,len(G_s)), np.repeat(G_s,len(C_s))])
		print parameters[77]


		clf = svm.SVC(kernel='rbf',decision_function_shape='ovr')
		scores = list()
		scores_std = list()
		start = time.time()
		for p in parameters:
			clf.C,clf.gamma = p[0],p[1]
			this_scores = cross_val_score(clf, pca_data[:,:-1],pca_data[:,-1], n_jobs=2)
			scores.append(np.mean(this_scores))
			scores_std.append(np.std(this_scores))
		end = time.time()
		print (end-start)
		plt.figure(1, figsize=(10, 10))
		plt.clf()
		for j in range(10):
			plt.subplot(5,2,j+1)
			title = "Accuracy with Gamma {:.2e}".format(parameters[10*j,1])
			plt.title(title)
			step = 10*j
			next_step = 10*(j+1)
			# print parameters[step:next_step]
			# print scores[step:next_step]
			plt.semilogx(parameters[step:next_step,0], scores[step:next_step])
			plt.semilogx(parameters[step:next_step,0], np.array(scores[step:next_step]) + np.array(scores_std[step:next_step]), 'b--')
			plt.semilogx(parameters[step:next_step,0], np.array(scores[step:next_step]) - np.array(scores_std[step:next_step]), 'b--')
			locs, ylabels = plt.yticks()
			plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
			plt.ylabel('CV score')
			plt.xlabel('Parameter C')
		plt.tight_layout()
		plt.savefig("../Figures/crossval_score_{}{}.png".format("PCA",20*i), bbox_inches='tight')
		print "Maximum in Trial {} with Score {}".format(scores.index(np.max(scores))+1,np.max(scores))

# for j in range(15):
# 	np.random.shuffle(pca_data)
# 	trainX = pca_data[:int(np.ceil(.8 * data_points)), :-1]
# 	trainY = pca_data[:int(np.ceil(.8*data_points)),-1]
# 	testX = pca_data[int(np.ceil(.8*data_points)):,:-1]
# 	testY = pca_data[int(np.ceil(.8*data_points)):,-1]
# 	clf.fit(trainX,trainY)
# 	print "Trial {} C: {} Gamma: {}\nScore".format(j+1, clf.C, clf.gamma)
# 	print clf.score(testX,testY)
