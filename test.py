import numpy as np
import infogain
X = np.array([['a','a','a'], ['a','a',1], [1,'a','a'],[1,1,1]])
y = np.array([['a'],['a'],[1],[1]])
#y.shape = (4,1)
infogain.infoGain(X,y)