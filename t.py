import infogain5
import numpy as np
X = np.array([[0,0], [0,0], [1,0],[1,0]])
y = np.array([[0],[0],[1],[1]])
a, b =infogain5.infoGain(X,y,1)
print a