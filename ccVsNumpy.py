import sys
sys.path.insert(0, './Data/')

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate, Dropout, add, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from smartPickle import pickleHandle, smartBatch
import numpy as np
import sys
dataDir = "./Data/"
ph = pickleHandle(500, "", dataDir + "equalDimsKeySigs/", "keyStrokeSigEqualDims", ".pickle", "")
batch, num = ph.getBatch(0)
#print(batch.x.shape)
#raise MemoryErrortry:
try:
    osName = "_"+sys.argv[1]
except:
    osName = ""
slidingWindow_module = tf.load_op_library("./windowing_out"+osName+".so")
tf.SlidingWindow = slidingWindow_module.sliding_window
"""
**** HYPERPARAMETERS ****
"""
window_size = 100
alpha = 0.01
"""
**** HYPERPARAMETERS ****
"""

def slidingWindow(inputs):
    outputs = []
    for inputArr in inputs:
        #inputArr = inputArr.reshape(inputArr.shape[1],)
        outputArr = []
        i = 0
        while i <= inputArr.shape[0] - window_size:
            outputArr.append(inputArr[i : i + window_size])
            i += 1
        outputs.append(outputArr)
    outputs = np.array(outputs)
    return outputs.astype(np.float32)

def windowLayer(inp):
    windowedSig = tf.py_func(slidingWindow, [inp], tf.float32)
    return windowedSig

def windowLayer_cc(inp):
    windowedSig = tf.SlidingWindow(inp, [window_size])
    return windowedSig

inputSig = Input((10000,))
ws = Lambda(windowLayer, output_shape=(9901, 100,))
output = ws(inputSig)
model = Model([inputSig], output)

inputSig1 = Input((10000,))
ws1 = Lambda(windowLayer_cc, output_shape=(9901, 100,))
output1 = ws1(inputSig1)
model2 = Model([inputSig1], output1)

from time import time

then = time()
out = model.predict(batch.x)
print(out)
print(out.shape)
print(time() - then)

then = time()
out = model2.predict(batch.x)
print(out)
print(out.shape)
print(time() - then)
