from collections import defaultdict
from sklearn import preprocessing
import numpy as np
import math
import operator
def infoGain(X, y, n = 2):
    """Compute chi-squared stats between each non-negative feature and class.

    This score can be used to select the n_features features with the
    highest values for the test chi-squared statistic from X, which must
    contain only non-negative features such as booleans or frequencies
    (e.g., term counts in document classification), relative to the classes.

    Recall that the chi-square test measures dependence between stochastic
    variables, so using this function "weeds out" the features that are the
    most likely to be independent of class and therefore irrelevant for
    classification.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = (n_samples, n_features_in)
        Sample vectors.

    y : array-like, shape = (n_samples,)
        Target vector (class labels).

    Returns
    -------
    chi2 : array, shape = (n_features,)
        chi2 statistics of each feature.
    pval : array, shape = (n_features,)
        p-values of each feature.

    Notes
    -----
    Complexity of this algorithm is O(n_classes * n_features).

    See also
    --------
    f_classif: ANOVA F-value between labe/feature for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    """

    # XXX: we might want to do some of the following in logspace instead for
    # numerical stability.
    ###X = check_array(X, accept_sparse='csr')
    ###if np.any((X.data if issparse(X) else X) < 0):
    ###    raise ValueError("Input X must be non-negative.")

    #Y = preprocessing.LabelBinarizer().fit_transform(y)
    #if Y.shape[1] == 1:                         ### if two classes, then transform it into two column
    #    Y = np.append(1 - Y, Y, axis=1)
    sampleNum, featureSize = X.shape
    classNum = y.shape[0]   
    scoreVector = []
    #print "featuresize, sampleNum. classNum:", featureSize,sampleNum, classNum
    #myMap count all the classes of samples having same feature value
    #secondMap count the classes of each feature value
    for i in range (featureSize):
        subtotal = 0
        myMap = {}
        secondMap = defaultdict(dict)
        rowCount = 0
        for ele in X:
            if str(ele[i]) in myMap:
                myMap[str(ele[i])] += 1
                if str(y[rowCount]) in secondMap[str(ele[i])]:
                    secondMap[str(ele[i])][str(y[rowCount])] += 1
                else:
                    secondMap[str(ele[i])][str(y[rowCount])] = 1
            else:
                myMap[str(ele[i])] = 1
                secondMap[str(ele[i])][str(y[rowCount])] = 1
            rowCount += 1
        #for k in myMap:
        #    print k, myMap[k]
        for ele in myMap:
            tmp = 0.0
            weight = float(myMap[ele]) / sampleNum
            for sub_ele in secondMap[ele]:
                current = secondMap[ele][sub_ele] / float(myMap[ele])
                #print "current is " , current
                tmp += current * math.log(current,2)
            subtotal += tmp * weight
    #    print "information for this feature is ", subtotal
        subtotal = -subtotal
        scoreVector.append(subtotal)
    total = 0.0
    for i in scoreVector:
        total += i
    yMap = {}
    oldScore = 0.0
    for i in y:
        if str(i) in yMap:
            yMap[str(i)] += 1
        else:
            yMap[str(i)] = 1
    #for i in yMap:
    #    print i, yMap[i]
    for ele in yMap:
        current = float(yMap[ele]) / sampleNum
        oldScore += current * math.log(current, 2)
    oldScore = -oldScore
    diff = {}
    #print "oldScore is ", oldScore
    counter = 0
    for i in scoreVector:
        diff[oldScore - i] = counter
        counter += 1
    #diffVe.sort(reverse = True)
    sorted_diff = sorted(diff.items(), key = operator.itemgetter(0), reverse = True)
    print "we select features : "
    for i in range(n):
        print "#", sorted_diff[i][1]
    print "information gain vector is  ", sorted_diff
    ###observed = safe_sparse_dot(Y.T, X)          # n_classes * n_features

    ###feature_count = check_array(X.sum(axis=0))
    ###class_prob = check_array(Y.mean(axis=0))
    ###expected = np.dot(class_prob.T, feature_count)

    ###return _chisquare(observed, expected)
