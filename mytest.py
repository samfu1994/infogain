#import numpy as np
#import infoGain
#X = np.array([[0,0,0], [0,0,1], [1,0,0],[1,1,1]])
#y = np.array([[0],[0],[1],[2]])
import numpy as np
import infogain3
import infogain2
import infogain5
import time
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
dest_num = 20
test_time = 10
iris = load_digits()
X, y = iris.data, iris.target
def mytest_info():
	#iris = load_iris()
	#print "X is ", X.shape
	t1 = time.clock()
	ch2 = SelectKBest(infogain5.infoGain, k = dest_num)
	X_comp = ch2.fit_transform(X,y)
	result = []
	for i in ch2.get_support(indices=True):
		result.append(i)
	t2 = time.clock()
	slot = t2 - t1
	#print "time is ", slot
	return result, slot

def mytest_chi2():
	#iris = load_iris()
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
v_chi2 = []
v_info = []
v_overlap = []
v_time = []
for current_time in range(test_time):
	result_chi2 , t_chi2 = mytest_chi2()
	#print "chi2: ", result_chi2
	v_chi2.append(t_chi2)
	result_info , t_info = mytest_info()
	v_info.append(t_info)
	#print "infogain: ", result_info
	overlap = 0
	num = len(result_info)
	for i in result_chi2:
		for j in result_info:
			if i == j:
				overlap += 1
				break
	v_overlap.append(overlap)
	v_time.append(t_info/t_chi2)
	#result_info = result_info.sort()
	#result_chi2 = result_chi2.sort()
print "chi2:     "
print np.mean(v_chi2)
print "infogain: "
print np.mean(v_info)
print "overlapped :", np.mean(v_overlap) , "/", num 
print np.mean(v_time) ," times slower"
