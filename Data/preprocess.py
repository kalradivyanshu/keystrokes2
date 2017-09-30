def onehot(s):
    labels = 51
    onehotVec = [0]*51
    onehotVec[s] = 1
    return onehotVec

import numpy as np
import pandas
import sklearn.preprocessing
import pickle

df = pandas.read_excel("data.xls")
keys = ['subject', 'sessionIndex', 'rep', 'H.period', 'UD.period.t', 'H.t',
       'UD.t.i', 'H.i', 'UD.i.e', 'H.e', 'UD.e.five', 'H.five',
       'UD.five.Shift.r', 'H.Shift.r', 'UD.Shift.r.o', 'H.o', 'UD.o.a', 'H.a',
       'UD.a.n', 'H.n', 'UD.n.l', 'H.l', 'UD.l.Return', 'H.Return']
sub = 0
mapSub = {}
x, y = [], []
print("Reading Data...")
for index, row in df.iterrows():
    subject, _, __, a, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20 = [row[key] for key in keys]
    try:
        s = mapSub[subject]
    except:
        mapSub[subject] = onehot(sub)
        s = mapSub[subject]
        sub += 1
    x.append([a, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20])
    y.append(s)
x = np.array(x)
print(sub)
print(x.shape)
y = np.array(y)
print(y.shape)
print(y[0])
print("Normalizing...")
x_new = sklearn.preprocessing.normalize(x)
print(x_new.shape)
print("Shuffling...")
rng_state = np.random.get_state()
np.random.shuffle(x_new)
np.random.set_state(rng_state)
np.random.shuffle(y)
np.random.set_state(rng_state)
np.random.shuffle(x)
print(x_new.shape)
print(y[0])
print(y.shape)
data = {}
data['x'] = x_new
data['y'] = y
data['x_old'] = x
print("Pickling...")
pickle.dump(data, open("keyStroke.pickle", "wb"))
