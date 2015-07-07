from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import infogain2
#X = np.array([['a','a','a'], ['a','a',1], [1,'a','a'],[1,1,1]])
#y = np.array([['a'],['a'],[1],[2]])
iris = load_digits()
X, y = iris.data, iris.target
ch2 = SelectKBest(infogain2.infoGain, k = 20)
X_comp = ch2.fit_transform(X,y)
result = []
for i in ch2.get_support(indices=True):
	result.append(i)
print X_comp.shape