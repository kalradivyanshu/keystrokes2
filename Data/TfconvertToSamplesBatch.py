import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate, Dropout, add, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from smartPickle import pickleHandle, smartBatch
import numpy as np
import pickle
import os
import sys

dataDir = "./"
ph = pickleHandle(250, "", dataDir + "equalDimsKeySigs/", "keyStrokeSigEqualDims", ".pickle", "")
batch, num = ph.getBatch(0)

try:
    osName = "_"+sys.argv[1]
except:
    osName = ""
slidingWindow_module = tf.load_op_library("./windowing_out"+osName+".so")
tf.SlidingWindow = slidingWindow_module.sliding_window

def windowLayer_cc(inp):
    window_size = 100
    windowedSig = tf.SlidingWindow(inp, [window_size])
    return windowedSig

inputSig = Input((10000,))
ws = Lambda(windowLayer_cc, output_shape=(9901, 100,))
output = ws(inputSig)
model = Model([inputSig], output)
j = 20400
while num != -1:
    print(j/20400, "%              ", end = "\r")
    j += batch.x.shape[0]
    samples = model.predict(batch.x)
    #print(samples.shape)
    y = batch.y
    samBat = smartBatch(samples, y, num)
    if not os.path.exists("./KeystrokeSigWindowedBatches"):
        os.makedirs("./KeystrokeSigWindowedBatches")
    pickle.dump(samBat, open("./KeystrokeSigWindowedBatches/finalIn_"+str(num)+".pickle", "wb"))
    batch, num = ph.getBatch(num)
