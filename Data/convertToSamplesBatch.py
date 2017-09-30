from smartPickle import pickleHandle, smartBatch
import numpy as np
import os
import pickle

pHan = pickleHandle(100)

d = []
j = 0
total = 20400
batch, num = pHan.getBatch()
while num != -1:
    xSig = batch.x
    y_sig = batch.y
    d = []
    for sig in xSig:
        if j % 5 == 0:
            print(j/total, end = "\r")
        i = 0
        samples = []
        for i in range(len(sig) - 100):
            sam = sig[i : i + 100]
            samples.append(sam)
        d.append(samples)
        j += 1
    sampleBatch = smartBatch(np.array(d), y_sig, num)
    if not os.path.exists("./SamplesBatch"):
        os.makedirs("./SamplesBatch")
    pickle.dump(sampleBatch, open("./SamplesBatch/sampleSignals_" + str(num) + ".pickle", "wb"))
    batch, num = pHan.getBatch(num)
dataDump = {}
