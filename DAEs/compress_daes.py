import pickle
import os
from pylearn2.utils import serial
import glob
from os.path import basename

daes = list()
labels = list()

for f in glob.glob(os.environ['MMDAEdaes'] + '*_best.pkl'):
    daes.append(serial.load(f))
    labels.append(basename(f))

fo = open(os.environ['MMDAEdaes'] + 'daes.pkl','wb')
pickle.dump([daes,labels], fo)
fo.close()
