import numpy as N
import os
import sys

try:
    data = N.load(os.environ['MMDAEdata'] + '/datasets.npz')
except IOError as e:
    print '\n\t' + os.environ['MMDAEdata'] + '/datasets.npz missing... Download as per instructions.\n'
    sys.exit()

keys = data.keys()

for key in keys:
    print "Saving %s ..." % key
    N.save(os.environ['MMDAEdata'] + '/' + key + '.npy',data[key])
    print "... done."

try:
    print "Deleting compressed file ..."
    os.remove(os.environ['MMDAEdata'] + '/datasets.npz')
    print "... done."
except:
    print "Compressed file not removed."
