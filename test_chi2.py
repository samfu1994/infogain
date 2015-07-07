import numpy as np
import time
import infoGain
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def mytest_chi2():
	dest_num = 3
	np.set_printoptions(threshold=dest_num + 1)
	iris = load_digits()
	#iris = load_iris()
	X, y = iris.data, iris.target
	#print "X size is ", X.shape
	#print "y size is ", y.shape
	#X = np.array([['a','a','a'], ['a','a',1], [1,'a','a'],[1,1,1]])
	#y = np.array([['a'],['a'],[1],[2]])
	t1 = time.clock()
	ch2 = SelectKBest(chi2, k = dest_num)
	X_comp = ch2.fit_transform(X,y)
	t2 = time.clock()
	result = []
	for i in ch2.get_support(indices=True):
		result.append(i)
	#print X_comp
	slot = t2-t1
	return result, slot