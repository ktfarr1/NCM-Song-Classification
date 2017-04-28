import numpy as np

'''
Constants
'''

song_offset = [0,0.8,0.279,0.15]
song_length = 2.0

'''
Helper functions that construct numpy arrays used by the various models
'''

def array_from_raw_data(directory,path_to_memory,path_to_keys,labels,bin_length):
	'''
	bin_length: length of bin in milliseconds
	'''
	spikes = np.loadtxt(path_to_memory)
	keys = np.loadtxt(path_to_keys,delimiter=',')
	n_bins = int(np.ceil(2000/bin_length))
	data = np.zeros((80,n_bins+1))
	count = 0
	for j in range(4):
		label = keys[np.where(keys==j)[0]]
		K = label.shape[0]
		for k in range(K):
			lower = label[k][0] + song_offset[j]
			upper = label[k][0] + song_length
			temp = (spikes[np.where((spikes>=lower) & (spikes <= upper))[0]]-lower)
			count = count +1
			data[(20*j)+k,n_bins] = j
			for i in range(n_bins):
				data[(20*j)+k,i] = len(np.where((temp>=i*(bin_length/1000.0)) & (temp<(i+1)*(bin_length/1000.0)))[0])

	# np.random.seed(0)
	# np.random.shuffle(data)
	# np.save("../Data/Numpy/{}/{}/{}_{}_{}.npy".format(directory,bin_length,labels[0],labels[1],bin_length),data)
	return data

def stack_arrays(directory,mem_paths,key_paths,labels,bin_length):
	if(key_paths.shape != mem_paths.shape):
		print "Keys and Memory do not have the same shape"
		return
	else:
		n_trials = key_paths.shape[0]
		data = np.empty((n_trials*80, int(np.ceil(2000/bin_length)+1)))
		for n in range(n_trials):
			data[(80 * n):((n + 1) * 80)] = array_from_raw_data(directory,mem_paths[n],key_paths[n],labels[n],bin_length)
		np.random.seed(0)
		np.random.shuffle(data)
		return data