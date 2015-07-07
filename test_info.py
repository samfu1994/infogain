#import numpy as np
#import infoGain
#X = np.array([['a','a','a'], ['a','a',1], [1,'a','a'],[1,1,1]])
#y = np.array([['a'],['a'],[1],[2]])
import numpy as np
import infoGain
import time
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def mytest_info(void):
	dest_num = 3
	np.set_printoptions(threshold= dest_num + 1)
	iris = load_digits()
	#iris = load_iris()
	X, y = iris.data, iris.target
	#print "X is ", X.shape
	t1 = time.clock()
	result = infoGain.infoGain(X,y,dest_num)
	t2 = time.clock()
	slot = t2 - t1
	return result,slot