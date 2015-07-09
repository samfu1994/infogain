from collections import defaultdict
from sklearn import preprocessing
from array import *
import numpy as np
import math
import operator
def infoGain(X, y, n = 2):
    sampleNum, featureSize = X.shape #get the parameters
    scoreVector = []
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    classNum = Y.shape[1]   
    X_T = np.transpose(X)
    for i in range (featureSize): #travel by columns
        subtotal = 0
        myMap = {} #myMap is to record the possible value of a particular feature, and how many times this value shows
        yMap = {}
        different_value_feature = 0
        current_feature = 0
        #the value of the map: how many times this class shows in this feature value in this particular feature 
        rowCount = 0
        yy = np.array([0] * sampleNum)
        different_value_feature = len(np.unique(X_T[i]))
        for j in range(sampleNum):
            if str(X[j][i]) not in myMap:
                myMap[str(X[j][i])] = different_value_feature
                different_value_feature += 1
            tmp_count = 0
            for k in Y[j]:
                if k != 1:
                    tmp_count += 1
                else:
                    yy[j] = tmp_count
                    if tmp_count in yMap:
                        yMap[tmp_count] += 1
                    else:
                        yMap[tmp_count] = 1
                    break
        mat = np.array([[0] * classNum] * different_value_feature)
        for j in range (sampleNum): #travel by rows
            index = myMap[str(X[j][i])] #if the class has been recorded before
            mat[index][yy[j]] += 1
        #for k in myMap:
        #    print k, myMap[k]
        for ele in myMap: #for each feature value
            tmp = 0.0
            current_index = myMap[ele]
            subSum = sum(mat[current_index])
            weight = float(subSum) / sampleNum
            for sub_ele in mat[current_index]: #for each class shows in this feature value
                if int(sub_ele) == 0:
                    continue
                current =  int(sub_ele) / float(subSum)#(p*log(p))
                #print "current is " , current
                tmp += current * math.log(current,2) #sigma(p*log(p))
            subtotal += tmp * weight#subtotal is the information gain of each feature
        subtotal = -subtotal
        scoreVector.append(subtotal)
    
    total = 0.0
    for i in scoreVector:
        total += i #sum up the values of all features
    oldScore = 0.0
    for ele in yMap:#calculate the value of father, will be subtracted by the sum thus get the information gain
        current = float(yMap[ele]) / sampleNum
        oldScore += current * math.log(current, 2)
    oldScore = -oldScore
    diff = array('d')
    empty = array('L')
    for i in scoreVector: #return value
        diff.append(oldScore - i)
        empty.append(0) #place holder
    return diff, empty

