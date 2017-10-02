import pickle
import numpy as np
dataDump = pickle.load(open('./keyStroke.pickle', 'rb'))
data, data_labels = dataDump['x_old'], dataDump['y']
signal = [3, 0, 5, 0, 7, 0, 11, 0, 13, 0, 17, 0, 19, 0, 23, 0, 29, 0, 31, 0, 37]
xSig = []
k = 0
for d in data:
    if k%100 == 0:
        print(k/204, "%", end = '\r')
    buf = 0
    sig = []
    for i in range(len(d)):
        time = int(d[i]*10000)
        if time >= 0:
            time = time - buf
            buf = 0
            [sig.append(signal[i]) for t in range(time)]
        else:
            buf = abs(time)
            last = len(sig)
            start = last - buf
            for j in range(start, last):
                sig[j] += signal[i + 1]
    sig += [0]*(len(sig)%100)
    #print(sig.shape)
    xSig.append(sig)
    k += 1
xSig = np.array(xSig)
dataDump = {}
dataDump['y'] = data_labels
dataDump['xSig'] = xSig
pickle.dump(dataDump, open("./keyStrokeSig.pickle", "wb"))
