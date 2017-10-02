import pickle
import numpy as np
from smartPickle import pickleHandle, smartBatch
import os

pHan = pickleHandle(250)

final = 20400
windowSize = 10000 # 1 second of keystrokes inserted into the nn.
X = []
Y = []
j = 0
batch, num = pHan.getBatch()
while num != -1:
    xSig = batch.x
    yLabel = batch.y
    X = []
    Y = []
    for sig, y in zip(xSig, yLabel):
        if j%10 == 0:
            print(j/final, end = '\r')
        sig = sig + [0]*([0 if not len(sig)%windowSize else windowSize - len(sig)%windowSize][0])
        sig = np.array(sig)
        sig = sig.reshape(sig.shape[0]//windowSize, windowSize)
        for signal in sig:
            X.append(signal)
            Y.append(y)
        j += 1
    X = np.array(X)
    Y = np.array(Y)
    sampleBatch = smartBatch(X, Y, num)
    if not os.path.exists("./equalDimsKeySigs"):
        os.makedirs("./equalDimsKeySigs")
    pickle.dump(sampleBatch, open("./equalDimsKeySigs/keyStrokeSigEqualDims" + str(num) + ".pickle", "wb"))
    batch, num = pHan.getBatch(num)
