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

    def __init__(self, batchSize = 500, pickleFile = "./keyStrokeSig.pickle", dirPh = "./smartData/", name = "keyStrokeSig", ext = ".smartPickle", separator = "_"):
        self.pickleFile = pickleFile
        self.dir = dirPh
        self.name = name
        self.ext = ext
        self.batchSize = batchSize
        self.separator = separator

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
            pickle.dump(bat, open(self.dir + self.name + self.separator + str(j) + self.ext, "wb"))
            j += 1
            i = i + self.batchSize
        del dataDump
        return True

    def getBatch(self, offset = 0, nxt = +1):
        try:
            f = open(self.dir + self.name + self.separator + str(offset + nxt) + self.ext, "rb")
        except:
            try:
                f = open(self.dir + self.name + self.separator + "1" + self.ext, "rb")
                f = None
                return [], -1
            except:
                self.createBatch()
                f = []
        finally:
            try:
                if f == None:
                    return [], -1
                del f
            except:
                pass
            f = open(self.dir + self.name + self.separator + str(offset + nxt) + self.ext, "rb")
        return pickle.load(f), offset + nxt
