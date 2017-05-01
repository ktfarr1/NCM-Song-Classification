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

def plot_confusion_matrix(matrix, classes, normalize, title):
    plt.imshow(matrix, interpolation='nearest',cmap='winter')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    print matrix
    threshold = matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]),range(matrix.shape[1])):
        plt.text(j, i , matrix[i,j], horizontalalignment='center',color='white' if matrix[i,j] > threshold else "black")
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

def plot_histograms():

    return



# if __name__ == '__main__': # Main function
#     wav_file = '../Data/CON12.wav' # Filename of the wav file
#     graph_spectrogram(wav_file)

# print wavfile.read('../Data/CON32.wav')