import numpy as np
import glob
import DataProcessing
from sklearn import model_selection

# for b in [2,4,8,16,32,64,128,256]:
# # for b in [1]:
#     for p in [150820,155665,160510]:
#     # for p in [155665]:
#         key_paths = np.array(glob.glob("../Data/{}/Keys/*.csv".format(p)))
#         mem_paths = np.array(glob.glob("../Data/{}/Memory/*.txt".format(p)))
#         labels = np.asarray([path.lstrip("..\\Data\\{}\\Keys\\".format(p)).upper().split('_')[0:2] for path in key_paths])
#         data = DataProcessing.stack_arrays(p,mem_paths,key_paths,labels,b)
#
#         np.save("../Data/{}_{}.npy".format(p,b),data)

# for b in [2,4,8,16,32,64,128,256]:
#     list = glob.glob("../Data/*_{}.npy".format(b))
#     print list
#     full = np.vstack([np.load(l) for l in list])
#     np.save("../Data/All_{}.npy".format(b),full)
#     # print np.sum(full)

for b in [2,4,8,16,32,64,128,256]:
    data = np.load("../Data/All_{}.npy".format(b))
    Xtr, Xte, Ytr, Yte = model_selection.train_test_split(data[:,:-1],data[:,-1], test_size=0.4, train_size=0.6, random_state=0, stratify=data[:,-1])
