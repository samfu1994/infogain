from collections import defaultdict
from sklearn import preprocessing
from array import *
import numpy as np
import math
import operator
def infoGain(X, y, n = 2):
    sampleNum, featureSize = X.shape #get the parameters
    classNum = y.shape[0]   
    scoreVector = []
    for i in range (featureSize): #travel by columns
        subtotal = 0
        myMap = {} #myMap is to record the possible value of a particular feature, and how many times this value shows
        mapIndex = {}
        different_value_class = 0
        secondMap = defaultdict(dict)#secondMap: first dimention: the feature value, second dimention: the class 
        #the value of the map: how many times this class shows in this feature value in this particular feature 
        rowCount = 0
        for j in range (sampleNum): #travel by rows
            if str(X[j][i]) in myMap: # the value in this feature has been recorded before
                myMap[str(X[j][i])] += 1  
                if str(y[rowCount]) in secondMap[str(X[j][i])]: #if the class has been recorded before
                    secondMap[str(X[j][i])][str(y[rowCount])] += 1
                else:
                    secondMap[str(X[j][i])][str(y[rowCount])] = 1
            else:
                myMap[str(X[j][i])] = 1
                secondMap[str(X[j][i])][str(y[rowCount])] = 1
            rowCount += 1
        #for k in myMap:
        #    print k, myMap[k]
        for ele in myMap: #for each feature value
            tmp = 0.0
            weight = float(myMap[ele]) / sampleNum
            for sub_ele in secondMap[ele]: #for each class shows in this feature value
                current = secondMap[ele][sub_ele] / float(myMap[ele])#(p*log(p))
                tmp += current * math.log(current,2) #sigma(p*log(p))
            subtotal += tmp * weight#subtotal is the information gain of each feature
        subtotal = -subtotal
        scoreVector.append(subtotal)
    total = 0.0
    for i in scoreVector:
        total += i #sum up the values of all features
    yMap = {}
    oldScore = 0.0
    for i in y:
        if str(i) in yMap:
            yMap[str(i)] += 1
        else:
            yMap[str(i)] = 1
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

