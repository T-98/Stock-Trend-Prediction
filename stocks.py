import numpy as np
from numpy import genfromtxt
from sklearn import svm 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


dataset = genfromtxt('finalData.csv',delimiter=',')
feat = dataset[:,:6]
output = dataset[:,7,None]
feat,output = shuffle(feat,output,random_state=10)
end = 1300
tr_end = 700
dele = np.zeros(0)
for i in range(feat.shape[0]):
    if i<end:
        vat = np.isnan(feat)
        val = vat[i]
        if val[4] == True:
            dele = np.append(dele,i)
nfeat = feat
j=0
for i in dele:
    nfeat = np.delete(nfeat,i-j,axis=0)
    j=j+1

#for i in range(nfeat.shape[0]):
#   v = np.isnan(nfeat)
#   if v[i,4] == True:
#       print 'found'
#   else:
#       print 'not found'
b = np.isinf(nfeat)
dele = np.zeros(0)
for i in range(nfeat.shape[0]):
    for j in range(nfeat.shape[1]):
        if b[i,j] == True:
            dele = np.append(dele,i)
            
j=0
for i in dele:
    nfeat = np.delete(nfeat,i-j,axis=0)
    j=j+1
    
standard_scaler = StandardScaler()
feat = standard_scaler.fit_transform(nfeat)
X_train = feat[10:tr_end,:]
Y_train = output[10:tr_end,:]
X_test = feat[1:10,:]
Y_test = output[1:10,:]
holdout = feat[tr_end:feat.shape[0],:]
holdoutY = output[tr_end:output.shape[0],:]
clf = svm.SVC(kernel = 'rbf',degree=9)
clf.fit(X_train,Y_train)
predicted =  clf.predict(X_test)
"""print holdout.shape
print holdoutY.shape
predicted = clf.predict(holdout)
print accuracy_score(holdoutY,predicted)"""
print ('accuracy=',accuracy_score(Y_test,predicted)*100, '%')


