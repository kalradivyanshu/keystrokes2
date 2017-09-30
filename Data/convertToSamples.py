import pickle
dataDump = pickle.load(open("./keyStrokeSig.pickle", "rb"))
data_labels = dataDump['y']
xSig = dataDump['xSig']
d = []
j = 0
final = len(xSig)
for sig in xSig:
    if j % 5 == 0:
        print(j/final, end = "\r")
    i = 0
    samples = []
    for i in range(len(sig) - 100):
        sam = sig[i : i + 100]
        samples.append(sam)
    d.append(samples)
    j += 1
dataDump = {}
dataDump['xSam'] = np.array(d)
dataDump['y'] = data_labels
print("Saving...")
pickle.dump(dataDump, open("./keyStrokeSigSample.pickle", "wb"))
