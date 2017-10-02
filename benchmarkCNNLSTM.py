import tensorflow as tf
from keras.models import Sequential, Model
import keras.layers
from keras.layers import Input, Dense, concatenate, Dropout, add, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import LSTM

import sys
sys.path.insert(0, './Data/')
from smartPickle import pickleHandle, smartBatch
import numpy as np

Conv1D = keras.layers.convolutional.Conv1D
dataDir = "./Data/"
ph = pickleHandle(250, "", dataDir + "KeystrokeSigWindowedBatches/", "finalIn", ".pickle", "_")
batch, num = ph.getBatch(0)
#print(batch.x.shape)
#raise MemoryError

"""
**** HYPERPARAMETERS ****
"""
alpha = 0.01
epochs = 1000
cnn1 = 50
cnn2 = 30
lstmOutputDim = 40
fc = [44, 60]
cnnAct = "relu"
fcAct = "tanh"
batch_size = 128

"""
**** HYPERPARAMETERS ****
"""

adam = Adam(alpha)
windowed = Input((100, 9901,))
convolutedSig = Conv1D(cnn1, 100, activation = cnnAct)
convolutedSig2 = Conv1D(cnn2, 100, activation = cnnAct)
cnn1Out = convolutedSig(windowed)
cnn2Out = convolutedSig2(cnn1Out)
#1511x9901x30
lstmOutput = LSTM(lstmOutputDim)(cnn2Out)
fullyConnected = Sequential()
fullyConnected.add(Dense(fc[0], input_dim = lstmOutputDim, activation = fcAct))
fullyConnected.add(Dense(fc[1], activation = fcAct))
fullyConnected.add(Dense(51, activation = 'softmax'))
output = fullyConnected(lstmOutput)
model = Model([inputSig], output)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

try:
    for i in range(epochs):
        while num != -1:
            print(i, "epoch ", num)
            model.fit([batch.x], batch.y, batch_size = batch_size, validation_split = 0.4, epochs = 1)
            batch, num = ph.getBatch(num)
except:
    pass
finally:
    model.save_weights('cnnlstmKeystrokes.h5')
