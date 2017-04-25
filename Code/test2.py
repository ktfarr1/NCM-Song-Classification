import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

spikes = np.loadtxt('../Data/AK EMEK single unit 150820-selected/EMEK051_unitB_Memory.txt')
keys = np.loadtxt('../Data/AK EMEK single unit 150820-selected/EMEK051_unitB_key.csv',delimiter=',')	
songlength = np.asarray([1.736, 3.689, 2.725, 2.218])

# print label[19][0]
# for k in label:
	# temp = np.where((spikes>=k[0]) & (spikes <= k[0]+songlength[int(k[1])]))
for j in range(4):
	f = plt.figure(figsize=(16,10))
	label = keys[np.where(keys==j)[0]]
	# print "This is label %d"%(j)
	# print label
	K = label.shape[0]
	binsize = int(np.ceil((songlength[j]*1000)/16))
	# print "Songlength(ms): %d, Bin Size: %d"%(songlength[j],binsize)
	for k in range(K):
		lower = label[k][0]
		upper = label[k][0]+songlength[j]
		print j,lower,upper
		temp = np.where((spikes>=lower) & (spikes <= upper))[0]
		# print spikes[temp]
		plt.subplot(4,5,k+1)
		# f.add_subplot(4,5,k+1)
		n, bins, patches = plt.hist(spikes[temp]-lower,binsize,normed=0, facecolor='green', alpha=0.75)
		if(int(label[k][1])>0):
			plt.title('CON%d Trial %d'%(int(label[k][1]),k+1),fontsize=5)
		else:
			plt.title('White Noise Trial %d'%(k+1),fontsize=5)
		plt.xlabel('Time',fontsize=4)
		plt.ylabel('Frequency',fontsize=4)
		plt.gca().set_xticklabels([])
		plt.gca().set_xticks([.25,.5,.75,1.0,1.25,1.5,1.75,2.0,2.25])
		plt.gca().set_yticklabels([])
	
	plt.tight_layout()
	if(j>0):
		plt.savefig('../Figures/CON%d_histograms'%(j))
	else:
		plt.savefig('../Figures/whitenoise_histograms')
