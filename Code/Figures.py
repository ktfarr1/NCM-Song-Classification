import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import itertools
from sklearn.metrics import confusion_matrix

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 256  # Length of the windowing segments
    fs = 256    # Sampling frequency
    pxx, freqs, bins, im = plt.specgram(data, nfft,fs)
    plt.axis('off')
    plt.savefig('../Figures/con1.png',
                dpi=1000, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png 

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def plot_confusion_matrix(y_true,y_predicted, normalize, title):
    classes = ["WN","CON1","CON2","CON3"]
    matrix = confusion_matrix(y_true,y_predicted)
    np.set_printoptions(precision=2)
    fig = plt.figure()
    plt.imshow(matrix, interpolation='nearest',cmap='winter')
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    print matrix
    plt.colorbar()
    threshold = matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]),range(matrix.shape[1])):
        plt.text(j, i , "{0:.2f}".format(matrix[i,j]), horizontalalignment='center',color='white' if matrix[i,j] > threshold else "black")
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    fig.savefig("../Figures/{}_Confusion_Matrix".format(title))
    # plt.show()

def plot_score(scores,x_ticks):
    labels = ["Bin Size", "Score"]
    x_axis = np.arange(1, 10, dtype=np.int)
    plt.figure(1, figsize=(6, 4))
    plt.plot(x_axis, scores, 'or-', linewidth=3)
    plt.grid(True)
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    plt.title("Bin Size vs. Score")
    plt.xlim(-0.1, 10.1)
    plt.ylim(-0.1, 1.1)
    plt.gca().set_xticks(x_axis)
    plt.gca().set_xticklabels(x_ticks)
    plt.legend(labels, loc="best")
    plt.tight_layout()
    plt.savefig("../Figures/SVM_score_line_plot.png")

# def plot_histograms():
#
#     return
#
# def plot_spike_train():
#
#     return




    # for k in label:
    # temp = np.where((spikes>=k[0]) & (spikes <= k[0]+songlength[int(k[1])]))
    # for j in range(4):
    # 	f = plt.figure(figsize=(16,10))
    # 	label = keys[np.where(keys==j)[0]]
    # 	# print "This is label %d"%(j)
    # 	# print label
    # 	K = label.shape[0]
    # 	binsize = int(np.ceil((songlength[j]*1000)/16))
    # 	# print "Songlength(ms): %d, Bin Size: %d"%(songlength[j],binsize)
    # 	for k in range(K):
    # 		lower = label[k][0]
    # 		upper = label[k][0]+songlength[j]
    # 		print j,lower,upper
    # 		temp = np.where((spikes>=lower) & (spikes <= upper))[0]
    # 		# print spikes[temp]
    # 		plt.subplot(4,5,k+1)
    # 		# f.add_subplot(4,5,k+1)
    # 		n, bins, patches = plt.hist(spikes[temp]-lower,binsize,normed=0, facecolor='green', alpha=0.75)
    # 		if(int(label[k][1])>0):
    # 			plt.title('CON%d Trial %d'%(int(label[k][1]),k+1),fontsize=5)
    # 		else:
    # 			plt.title('White Noise Trial %d'%(k+1),fontsize=5)
    # 		plt.xlabel('Time',fontsize=4)
    # 		plt.ylabel('Frequency',fontsize=4)
    # 		plt.gca().set_xticklabels([])
    # 		plt.gca().set_xticks([.25,.5,.75,1.0,1.25,1.5,1.75,2.0,2.25])
    # 		plt.gca().set_yticklabels([])
    #
    # 	plt.tight_layout()
    # 	if(j>0):
    # 		plt.savefig('../Figures/CON%d_histograms'%(j))
    # 	else:
    # 		plt.savefig('../Figures/whitenoise_histograms')
