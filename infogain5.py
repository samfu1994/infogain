from collections import defaultdict
from sklearn import preprocessing
from array import *
from collections import Counter
import numpy as np
import numpy.ma as ma
import math
import operator
import time
import threading
def infoGain(X, y, n = 2):
	sampleNum, featureSize = X.shape #get the parameters
	scoreVector = []
	Y = preprocessing.LabelBinarizer().fit_transform(y)
	if Y.shape[1] == 1:                         ### if two classes, then transform it into two column
		Y = np.append(1 - Y, Y, axis=1)
	classNum = Y.shape[1]
	X_T = X
	X_T = np.transpose(X_T)
	diff = array('d')
	empty = array('L')
	symbol = 2147483647
	oldScore = 0.0
	X_T[X_T == 0] = symbol
	f = open("time", "w")
	for i in range(X_T.shape[0]):
		count_each_feature_value = Counter(X_T[i]) 
		feature_value = np.unique(X_T[i])
		num_value_feature = len(feature_value)
		subtotal = 0
		count = 0
		for ele in feature_value:# each value in this feature
			weight = count_each_feature_value[feature_value[count]] / float(sampleNum)
			after_mask = ma.masked_not_equal(X_T[i], ele)
			ma.set_fill_value(after_mask,0)
			after_mask = after_mask.filled()
			mat_each_value_feature = np.dot(after_mask, Y) #
			tmp = mat_each_value_feature
			tmp_sum = np.sum(mat_each_value_feature)
			mat_each_value_feature = np.dot(mat_each_value_feature,  1.0 / tmp_sum)
			current = 0
			#after_compress = ma.masked_equal(mat_each_value_feature, 0)
			#after_compress = after_compress.compressed()
			#t5 = time.clock()
			#current = np.dot(mat_each_value_feature, np.log2(after_compress))
			#t6 = time.clock()
			#mat_each_value_feature = mat_each_value_feature[-mat_each_value_feature.mask]
			#ma.set_fill_value(mat_each_value_feature,1)
			#mat_each_value_feature = mat_each_value_feature.filled()
			#current = np.dot(mat_each_value_feature, np.log2(mat_each_value_feature))
			#current = 0
			for k in mat_each_value_feature:
				if k == 0:
					continue;
				current += k * math.log(k,2)
			subtotal += current * weight
			count += 1
		subtotal = - subtotal
		diff.append(oldScore - subtotal)
		empty.append(0)
	return diff, empty