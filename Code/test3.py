import numpy as np
import glob
from fractions import Fraction
from sklearn import svm

# key_paths = np.array(glob.glob("..\\Data\\150820\\Keys\\*41*B*.csv"))
# mem_paths = np.array(glob.glob("..\\Data\\150820\\Memory\\*41*B*.txt"))
# print key_paths, mem_paths

song_offset = [0,0.8,0.279,0.15]
songlength = 2.0

def create_numpy_array(path_to_memory,path_to_keys,bin_length,id_number):
	'''
	bin_length: length of bin in milliseconds
	'''
	spikes = np.loadtxt(path_to_memory)
	keys = np.loadtxt(path_to_keys,delimiter=',')
	binsize = int(np.ceil(2000/bin_length))
	data = np.zeros((80,binsize+1))
	# print 2000/16.0
	for j in range(4):
		label = keys[np.where(keys==j)[0]]
		K = label.shape[0]
		for k in range(K):
			lower = label[k][0] + song_offset[j]
			upper = label[k][0] + songlength
			temp = (spikes[np.where((spikes>=lower) & (spikes <= upper))[0]]-lower)
			data[(20*j)+k,binsize] = j
			for i in range(binsize):
				data[(20*j)+k,i] = len(np.where((temp>=i*(bin_length/1000.0)) & (temp<2*i*(bin_length/1000.0)))[0])

	np.random.seed(0)
	np.random.shuffle(data)
	# np.save("../Data/Numpy/150820/150820_%d_%d.npy"%(bin_length,id_number),data)
	return data

# if(key_paths.shape[0] == mem_paths.shape[0]):
# 	R = key_paths.shape[0]
# 	for r in range(R):
# 		create_numpy_array(mem_paths[r],key_paths[r],16,r)
# 		# print mem_paths[r], key_paths[r]
# else:
# 	print "key_paths does not have the same shape as mem_paths"

# create_numpy_array(mem_paths[0],key_paths[0],2,666)

# data = np.vstack([np.load("../Data/Numpy/150820/150820_16_36.npy"),np.load("../Data/Numpy/150820/150820_16_37.npy"),np.load("../Data/Numpy/150820/150820_16_38.npy")])
# data = np.vstack([np.load("../Data/Numpy/150820/150820_16_6.npy"),np.load("../Data/Numpy/150820/150820_16_26.npy"),np.load("../Data/Numpy/150820/150820_16_16.npy")])
# data = np.load("../Data/Numpy/150820/150820_16_36.npy")
# test = np.vstack([np.load("../Data/Numpy/150820/150820_16_5.npy"),np.load("../Data/Numpy/150820/150820_16_25.npy"),np.load("../Data/Numpy/150820/150820_16_15.npy")])
# data = np.load("../Data/Numpy/150820/150820_2_666.npy")

data = np.empty((55*80,126))
for i in range(55):
	data[(80*i):((i+1)*80)] = np.load("../Data/Numpy/150820/150820_16_{}.npy".format(i))
data_points = data.shape[0]
n_features = data.shape[1]-1
# np.random.seed(97)

gamma = np.empty(5,)
gamma[:] = 1.0/n_features
gamma = np.power(gamma,np.arange(1,6))
c = [0.1,1.0,10.0,100.0,1000.0]
parameters = np.transpose([np.tile(c,len(gamma)), np.repeat(gamma,len(c))])
print parameters[4,0], (Fraction(parameters[7,1]).limit_denominator())
# best_score = 0
# best_weights = []
# for j in range(200):
# 	temp = np.copy(data)
# 	np.random.shuffle(temp)
# 	trainX = temp[:int(np.ceil(.8*data_points)),:-1]
# 	trainY = temp[:int(np.ceil(.8*data_points)),-1]
# 	testX = temp[int(np.ceil(.8*data_points)):,:-1]
# 	testY = temp[int(np.ceil(.8*data_points)):,-1]
# 	# clf = svm.SVC(C=parameters[j,0],class_weight='balanced',decision_function_shape='ovr',gamma=parameters[j,1])
# 	clf = svm.SVC(C=parameters[4,0],class_weight='balanced',decision_function_shape='ovr',gamma=parameters[7,1])
# 	clf.fit(trainX,trainY)
# 	score = clf.score(testX,testY)
# 	if(score>best_score):
# 		best_score = score
# 		best_weights = clf.dual_coef_
# 	print "{} out of 200".format(j+1)
# np.save('../Data/weights.npy',best_weights)
# np.save("../Data/accuracy.npy",accuracy)
'''
np.random.seed(97)
np.random.shuffle(data)
np.random.shuffle(test)
gamma = [1.0/1920,1.0/960,1.0/480,1.0/240]
# gamma = [1.0/640,1.0/320,1.0/160,1.0/80,1.0/40]
c = np.asarray([0.1,1.0,10.0,100.0,1000.0])*1
parameters = np.transpose([np.tile(c,len(gamma)), np.repeat(gamma,len(c))])
N = parameters.shape[0]
trainX = data[:216,:-1]
trainY = data[:216,-1]
testX = data[216:,:-1]
testY = data[216:,-1]

optimal_clf = svm.SVC()
optimal_score = 0

for params in parameters:
	clf = svm.SVC(C=params[0],gamma=params[1],class_weight='balanced',decision_function_shape='ovr')
	# clf = svm.SVC(C=params[0],gamma=params[1],decision_function_shape='ovr')
	clf.fit(trainX,trainY)
	score = clf.score(testX,testY)
	if(score > optimal_score):
		optimal_score = score
		optimal_clf = clf
	# print "C: {} Gamma: {} Score: {}".format(params[0],str(Fraction(params[1]).limit_denominator()),score)
# clf = svm.SVC(class_weight='balanced',decision_function_shape='ovr')
# clf.fit(trainX,trainY)
print Fraction(optimal_clf.gamma).limit_denominator()
# for t in test:
print optimal_clf.score(test[:,:-1],test[:,-1])
# print clf.score(testX,testY)
# print clf.get_params()
'''