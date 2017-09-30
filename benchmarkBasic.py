from CalculateEER import GetEER
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import numpy as np

print("Loading Data...")
dataDump = pickle.load(open('./Data/keyStroke.pickle', 'rb'))
data, data_labels = dataDump['x_old'], dataDump['y']
print('Dividing Data...')

train, train_labels = data[:int(20400*0.6)], data_labels[:int(20400*0.6)]
valid, valid_labels = data[int(20400*0.6):int(20400*0.8)], data_labels[int(20400*0.6):int(20400*0.8)]
test, test_labels = data[int(20400*0.8):], data_labels[int(20400*0.8):]

print(len(train), len(valid), len(test))
print('Divided data.')

def benchmarkTensorNN(N, Nodes):
    print(N, Nodes)
    if 0 in Nodes: return -1000
    input_size, num_labels, beta, batch_size, epochs = 21, 51, 10e-4, 150, 3000
    model = Sequential()
    model.add(Dense(Nodes[0], input_dim=input_size, kernel_regularizer=regularizers.l2(beta)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    for i in range(1, len(Nodes)):
        model.add(Dense(Nodes[i], kernel_regularizer=regularizers.l2(beta)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(num_labels, kernel_regularizer=regularizers.l2(beta)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    adam = Adam()
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    alphas = [0.01, 0.009, 0.005, 0.004, 0.001]
    #alphas = [alphas[0]]
    for alpha in alphas:
        print("\n")
        print("==============================================",alpha,"==============================================")
        print("\n")
        model.optimizer.lr.assign(alpha)
        model.fit(data, data_labels, batch_size = batch_size, validation_split = 0.4, epochs = epochs)
    validpredictions = model.predict_proba(valid)
    eer = GetEER(validpredictions, valid_labels)
    print("******************************************")
    print(eer)
    print("******************************************")
    return -1*eer

if __name__ == "__main__":
	EERs = list()
	for i in range(5):
		EERs.append(benchmarkTensorNN(5, [42, 50, 82, 70, 61]))
	print(EERs)
	print(np.std(np.array(EERs)))
	print(sum(EERs)/len(EERs))
