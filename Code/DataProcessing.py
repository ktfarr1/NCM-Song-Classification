import numpy as np

'''
Constants
'''

song_offset = [0,0.8,0.279,0.15]
song_length = 2.0

'''
Helper functions that construct numpy arrays used by the various models
'''
'''
This function transforms a single spike train into a usable numpy array

A spike train is a sequence of times at which an electrode detected firing in
the neuron currently being monitored, which is the file at path_to_memory.
The second file, path_to_keys, contains the time codes at which a labeled stimulus
was provided. In this case, the labels are [0,1,2,3] corresponding to:
	White Noise = 0
	CON1 		= 1
	CON2		= 2
	CON3		= 3

Directory and Labels are parsed from the name of the file, and allow you to save named
copies of the spike trains from individual cells rather than conglomerating them, but the
default is to return the array to be stacked with other cells.

Finally, the bin_length represents the duration in milliseconds that you would like to use
to count spike events. The maximum precision is 1ms, and this leads to extremely poor performance.

The method loads the data from file, creates an empty numpy array of the correct shape, then filters
out individual labels from the keys file. Next, it adds the label, and creates the bounds for each bin
Finally, it searches the data array for the spikes that belong to the bin, and subtracts the lower bound
so each row in the array is 0-2000 ms.
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

'''
stack_arrays is used by the main function to conglomerate many different spike trains into a single matrix
It loops over all the paths provided (typically by glob) and performs array_from_raw_data on each one.
Initially, I was stacking them with vstack (thus the name of the function), but I found it sped up when I initialized
an empty array and then filled it.

Here, directory and labels are arrays of labels, corresponding with the correct keys and memory files.
Bin_length is required here as well, to produce a uniform data set.
'''

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


def sum_to_one(input_array):
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