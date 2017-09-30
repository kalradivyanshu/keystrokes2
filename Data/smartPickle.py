import pickle
import os

class smartBatch:
    def __init__(self, x, y, num):
        self.x = x
        self.y = y
        self.batchNum = num
    def save(self, path):
        pickle.dump(self, open(path, "wb"))

class pickleHandle:

    def __init__(self, batchSize = 500):
        self.pickleFile = "./keyStrokeSig.pickle";
        self.dir = "./smartData/"
        self.name = "keyStrokeSig"
        self.ext = ".smartPickle"
        self.batchSize = batchSize

    def createBatch(self):
        print("Loading Data...")
        dataDump = pickle.load(open(self.pickleFile, "rb"))
        data, data_labels = dataDump['xSig'], dataDump['y']
        print("Done. Creating Batches...")
        final = len(data)
        i = 0
        j = 1
        while i < final - self.batchSize:
            if (j - 1)%3 == 0:
                print(100*i/final, "%                             ", end = '\r')
            batch_x = data[i : i + self.batchSize]
            batch_y = data_labels[i : i + self.batchSize]
            bat = smartBatch(batch_x, batch_y, j)
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            pickle.dump(bat, open(self.dir + self.name + "_" + str(j) + self.ext, "wb"))
            j += 1
            i = i + self.batchSize
        del dataDump
        return True

    def getBatch(self, offset = 0, nxt = +1):
        try:
            f = open(self.dir + self.name + "_" + str(offset + nxt) + self.ext, "rb")
        except:
            try:
                f = open(self.dir + self.name + "_" + "1" + self.ext, "rb")
                return [], -1
            except:
                self.createBatch()
        finally:
            del f
            f = open(self.dir + self.name + "_" + str(offset + nxt) + self.ext, "rb")
        return pickle.load(f), offset + nxt
