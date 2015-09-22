import pickle
import os
from pylearn2.utils import serial


try:
    fo = open(os.environ['MMDAEdaes'] + 'daes.pkl','rb')
    daes,labels = pickle.load(fo)
    fo.close()
except IOError as e:
    print '\n\t' + os.environ['MMDAEdaes'] + 'daes.pkl missing... Download as per instructions.\n'
    sys.exit()

for dae,label in zip(daes,labels):
    print "Saving %s ..." % label
    serial.save(os.environ['MMDAEdaes'] + label,dae)
    print "... done."

try:
    print "Deleting compressed file ..."
    os.remove(os.environ['MMDAEdaes'] + 'daes.pkl')
    print "... done."
except:
    print "Compressed file not removed."
