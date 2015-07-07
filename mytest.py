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
def mytest_info():
	dest_num = 20
	iris = load_digits()
	#iris = load_iris()
	X, y = iris.data, iris.target
	#print "X is ", X.shape
	t1 = time.clock()
	result = infoGain.infoGain(X,y,dest_num)
	t2 = time.clock()
	slot = t2 - t1
	#print "time is ", slot
	return result, slot

def mytest_chi2():
	dest_num = 20
	iris = load_digits()
	#iris = load_iris()
	X, y = iris.data, iris.target
	t1 = time.clock()
	ch2 = SelectKBest(chi2, k = dest_num)
	X_comp = ch2.fit_transform(X,y)
	result = []
	for i in ch2.get_support(indices=True):
		result.append(i)
	t2 = time.clock()
	slot = t2-t1
	result.sort()
	#print result
	#print "time is ", slot
	return result, slot
dd = load_digits()
xx = dd.data
print xx.shape
result_chi2 , t_chi2 = mytest_chi2()
print "chi2:     "
print result_chi2, t_chi2
result_info , t_info = mytest_info()
print "infogain: "
print result_info, t_info
overlap = 0
num = len(result_info)
for i in result_chi2:
	for j in result_info:
		if i == j:
			overlap += 1
			break
print "overlapped :", overlap , "/", num
print t_info / t_chi2 ," times slower"
#result_info = result_info.sort()
#result_chi2 = result_chi2.sort()
