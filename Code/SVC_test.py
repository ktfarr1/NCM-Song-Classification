import numpy as np
import glob
from fractions import Fraction
from sklearn import svm
import copy
from sklearn import preprocessing

# key_paths = np.array(glob.glob("..\\Data\\150820\\Keys\\*.csv"))
# mem_paths = np.array(glob.glob("..\\Data\\150820\\Memory\\*.txt"))
key_paths = np.array(glob.glob("../Data/150820/Keys/*022*A*.csv"))
mem_paths = np.array(glob.glob("../Data/150820/Memory/*022*A*.txt"))
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
	# np.random.seed(0)
	# np.random.shuffle(data)
	# np.save("../Data/Numpy/150820/150820_{}_{}.npy".format(bin_length,id_number),data)
	return data

def frequency_transform(input_array):
	rows = input_array.shape[0]
	output_array = np.empty(input_array.shape,dtype=np.float)
	for row in range(rows):
		element_sum = np.sum(input_array[row,:-1])
		if(element_sum != 0):
			output_array[row,:-1] = input_array[row,:-1]/float(element_sum)
			output_array[row,-1] = input_array[row,-1]
		else:
			return np.array(input_array,dtype=np.float)
	return output_array


# test = []
# if(key_paths.shape[0] == mem_paths.shape[0]):
# 	R = key_paths.shape[0]
# 	for r in range(R):
# 		test = create_numpy_array(mem_paths[r],key_paths[r],16,r)
# 		# print mem_paths[r], key_paths[r]
# else:
# 	print "key_paths does not have the same shape as mem_paths"

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
frequency_normalized = frequency_transform(data)


# print data[0]
# print data[:,:-1].shape
# print normalize(data[:,:-1],norm='l2',copy=True,axis=1)[0:3,-1]
# print data[0:2,0:2]
gamma = np.empty(5,)
gamma[:] = 1.0/n_features
gamma = np.power(gamma,np.arange(1,6))
c = [0.1,1.0,10.0,100.0,1000.0]
parameters = np.transpose([np.tile(c,len(gamma)), np.repeat(gamma,len(c))])
print parameters[4,0], (Fraction(parameters[7,1]).limit_denominator())
best_score = 0
# best_weights = []
best_clf = svm.SVC()
# for k in range(5)
for j in range(15):
	temp = np.copy(frequency_normalized)
	np.random.shuffle(temp)
	trainX = temp[:int(np.ceil(.8*data_points)),:-1]
	trainY = temp[:int(np.ceil(.8*data_points)),-1]
	testX = temp[int(np.ceil(.8*data_points)):,:-1]
	testY = temp[int(np.ceil(.8*data_points)):,-1]
	# clf = svm.SVC(C=parameters[j,0],class_weight='balanced',decision_function_shape='ovr',gamma=parameters[j,1])
	clf = svm.SVC(C=parameters[4,0],class_weight='balanced',decision_function_shape='ovr',gamma=parameters[7,1],probability=False)
	clf.fit(trainX,trainY)
	score = clf.score(testX,testY)
	if(score>best_score):
		best_score = score
		best_clf = copy.deepcopy(clf)
		# best_weights = clf.dual_coef_
	print "{} out of 50".format(j+1)
	print score
print "Best Score: {}".format(best_score)
print "Support vectors shape: {}".format(best_clf.support_vectors_.shape)
print "Support vectors per class: {}".format(best_clf.n_support_)
# np.random.shuffle(test)
# print ideal_clf.score(test[:,:-1],test[:,-1])

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