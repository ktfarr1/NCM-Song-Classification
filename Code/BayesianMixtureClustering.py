import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn import preprocessing

def sum_to_zero(input_array):
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

# Number of samples per component
n_samples = 1100
data = np.empty((55*80,126))
for i in range(55):
	data[(80*i):((i+1)*80)] = np.load("../Data/Numpy/150820/150820_16_{}.npy".format(i))
data_points = data.shape[0]
n_features = data.shape[1]-1
np.random.shuffle(data)
X = preprocessing.Normalizer(norm='l2',copy=True).fit_transform(data[:,:-1])



gmm = mixture.BayesianGaussianMixture(n_components=4, covariance_type='tied').fit(X)
predictions = gmm.predict(X)
heatmap, xedges, yedges = np.histogram2d(np.asarray(data[:,-1]),predictions,bins=(4,4))
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# print extent
# plt.clf()
print heatmap
print heatmap.shape
# print xedges
# print yedges
# x,y = np.meshgrid(xedges,yedges)
# print x
# print y
fig = plt.figure(figsize=(7,7))
plt.pcolormesh(np.arange(5),np.arange(5),heatmap,cmap="Blues")
plt.gca().set_xticklabels(["WN","CON1","CON2","CON3"])
plt.gca().set_xticks((np.arange(0,4)+0.5))
plt.gca().set_yticklabels(np.arange(0,4))
plt.gca().set_yticks((np.arange(0,4)+0.5))
plt.colorbar()
plt.show()
# plt.imshow(heatmap, extent)
# plt.scatter(data[:,-1],predictions)
# plt.figure(figsize=(8,8))
# for i in range(gmm.means_.shape[0]):
# 	plt.subplot(4,4,i+1)
# 	# plt.scatter(range(gmm.means_.shape[1]),gmm.means_[i])
# 	plt.plot(range(gmm.means_.shape[1]),gmm.means_[i])
# plt.show()
