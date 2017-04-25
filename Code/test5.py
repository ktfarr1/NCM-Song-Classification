import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import decomposition
from fractions import Fraction
from sklearn import svm
import copy
if __name__ == '__main__':
	data = np.empty((45*80,126))
	for i in range(45):
		data[(80*i):((i+1)*80)] = np.load("../Data/Numpy/150820/150820_16_{}.npy".format(i))
	data_points = data.shape[0]
	n_features = data.shape[1]-1
	np.random.seed(0)
	np.random.shuffle(data)
	labels = data[:,-1]
	samples = data[:,:-1]

	normed = preprocessing.Normalizer(norm='l2',copy=True).fit_transform(samples)
	pca = decomposition.PCA(n_components=20,copy=True, svd_solver='full').fit(normed)
	# pca_data = np.hstack([pca.transform(normed), labels.reshape(-1,1)])
	pca_data = data
	C_s = np.logspace(-5,5,10)
	G_s = np.logspace(-5,1,10,base=(pca_data.shape[0]-1))

	parameters = np.transpose([np.tile(C_s,len(G_s)), np.repeat(G_s,len(C_s))])
	print parameters[77]


	clf = svm.SVC(kernel='rbf',decision_function_shape='ovr')
	scores = list()
	scores_std = list()
	for p in parameters:
		clf.C,clf.gamma = p[0],p[1]
		this_scores = cross_val_score(clf, pca_data[:1000,:-1],pca_data[:1000,-1], n_jobs=1)
		scores.append(np.mean(this_scores))
		scores_std.append(np.std(this_scores))

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
		locs, labels = plt.yticks()
		plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
		plt.ylabel('CV score')
		plt.xlabel('Parameter C')
	plt.tight_layout()
	plt.savefig("../Figures/crossval_score_{}.png".format(1), bbox_inches='tight')
	print "Maximum in Trial {} with Score {}".format(scores.index(np.max(scores)),np.max(scores))

# for j in range(15):
# 	np.random.shuffle(pca_data)
# 	trainX = pca_data[:int(np.ceil(.8 * data_points)), :-1]
# 	trainY = pca_data[:int(np.ceil(.8*data_points)),-1]
# 	testX = pca_data[int(np.ceil(.8*data_points)):,:-1]
# 	testY = pca_data[int(np.ceil(.8*data_points)):,-1]
# 	clf.fit(trainX,trainY)
# 	print "Trial {} C: {} Gamma: {}\nScore".format(j+1, clf.C, clf.gamma)
# 	print clf.score(testX,testY)
