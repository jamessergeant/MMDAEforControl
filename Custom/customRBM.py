import numpy as N
np = N
import csv
import os.path
from os.path import basename, splitext
from theano.compat.six.moves import xrange
from pylearn2.datasets import control
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace, Conv2DSpace
from matplotlib import pyplot as plt
import theano
import time
import pickle

class CustomLoader(DenseDesignMatrix):

    def __init__(self, data_file, which_set, batch_size, sequence=1, normalise=None, shuffle=False,
                 start=None, stop=None, axes=['b', 0, 1, 'c']):

        self.args = locals()

        assert which_set in ['train', 'valid']

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        topo_view = N.load(data_file)

        if start is not None:
            assert stop is not None
            assert start >= 0
            assert stop > start
            assert ((stop-start)%batch_size) == 0
            if stop > topo_view.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            topo_view = topo_view[start:stop, :]
            if topo_view.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))

        if normalise == 1:
            topo_mean = topo_view.mean()
            topo_std = topo_view.std()
        elif normalise == 2:
            topo_mean = topo_view.mean(axis=0)
            topo_std = topo_view.std(axis=0)

        topo_view = (topo_view - topo_mean) / topo_std

        if sequence != 1:
            temp = topo_view
            topo_view = np.zeros([temp.shape[0],sequence,temp.shape[1]])
            for i in range(0,temp.shape[0]-sequence):
                topo_view[i,:,:] = temp[i:i+sequence,:].reshape(1,sequence,temp.shape[1])
            del temp
        else:
            topo_view = topo_view.reshape(topo_view.shape[0],1,topo_view.shape[1])

        m, r, c = topo_view.shape
        assert r == sequence
        topo_view = topo_view.reshape(m, r, c, 1)

        if shuffle:
            self.shuffle_rng = make_np_rng(
                None, [1, 2, 3], which_method="shuffle")
            for i in xrange(topo_view.shape[0]):
                j = self.shuffle_rng.randint(m)
                # Copy ensures that memory is not aliased.
                tmp = topo_view[i, :, :, :].copy()
                topo_view[i, :, :, :] = topo_view[j, :, :, :]
                topo_view[j, :, :, :] = tmp
                
        super(CustomLaser, self).__init__(topo_view=dimshuffle(topo_view))

        assert not N.any(N.isnan(self.X))

# For use in greedy layer-wise training of multi-modal deep layer RBM
class CustomMMPosterior(DenseDesignMatrix):

    def __init__(self, model_file1, data_file1, model_file2, data_file2, which_set, batch_size, shuffle=False,
                 start=None, stop=None, length=None,
                 axes=['b', 0, 1, 'c'],
                 preprocessor=None,
                 fit_preprocessor=False,
                 fit_test_preprocessor=False):

        self.args = locals()
        existing_data_path = '/home/n8382921/bin/BEB801/datasets/multimodal_hidrep/' + splitext(basename(model_file1))[0] + '_' + splitext(basename(data_file1))[0] + '_' + splitext(basename(model_file2))[0] + '_' + splitext(basename(data_file2))[0] +'.npy'
        model_files = [model_file1, model_file2]
        data_files = [data_file1, data_file2]
        assert which_set in ['train', 'valid']
        assert type(model_files) is list
        assert type(data_files) is list
        assert len(model_files) == len(data_files)

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        # only process if it hasn't been done already
        if not os.path.exists(existing_data_path):
            for ii in range(len(model_files)):

                if control.get_load_data():

                    # Load single mode model for first layer
                    model = serial.load(model_files[ii])
                    # Load the single mode data
                    data = N.load(data_files[ii])
                    sequence = model.dataset_yaml_src.split('sequence: ')
                    if len(sequence) > 1:
                        sequence = int(sequence[1].split(',')[0])
                    else:
                        sequence = 1
                    if 'sequence_old' in locals():
                        assert sequence == sequence_old
                    sequence_old = sequence
                    if sequence != 1:
                        temp = data
                        data = np.zeros([temp.shape[0],sequence*temp.shape[1]])
                        for i in range(0,temp.shape[0]-sequence):
                            data[i,:] = temp[i:i+sequence,:].reshape(1,sequence*temp.shape[1])
                        del temp
                    if start is not None:
                        assert stop is not None
                        assert start >= 0
                        assert stop > start
                        assert ((stop-start)%batch_size) == 0
                        if stop > data.shape[0]:
                            raise ValueError('stop=' + str(stop) + '>' +
                                             'm=' + str(self.X.shape[0]))
                        data = data[start:stop, :]
                        if data.shape[0] != stop - start:
                            raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                             % (self.X.shape[0], start, stop))

                    # Process data to get hidden representation from first layer
                    data = theano.shared(data)
                    data = model.mf(data)
                    data = data[0]
                    data = data[0]
                    data = data.eval()

                    if 'topo_view' not in locals():
                        topo_view = data.reshape(data.shape[0],1,data.shape[1])
                    else:
                        topo_view = N.append(topo_view, data.reshape(data.shape[0],1,data.shape[1]),axis=2)
                    print "laser finished"


                else:
                    # Does this get used? Would this even be entered due to lack of model_files, data_files?
                    print "data_loader CustomMMPosterior ELSE STATEMENT USED"
                    if which_set == 'train':
                        size = 10000
                    elif which_set == 'test':
                        size = 2000
                    elif which_set == 'valid':
                        size = 2000
                    else:
                        raise ValueError(
                            'Unrecognized which_set value "%s".' % (which_set,) +
                            '". Valid values are ["train","test"].')
                    topo_view = np.random.rand(size, 1, 2)

            m, r, c = topo_view.shape

            topo_view = topo_view.reshape(m, r, c, 1)

            # save the data to avoid reprocessing later
            N.save(existing_data_path, topo_view)
        else:
            topo_view = N.load(existing_data_path)
            m = topo_view.shape[0]

        if shuffle:
            self.shuffle_rng = make_np_rng(
                None, [1, 2, 3], which_method="shuffle")
            for i in xrange(topo_view.shape[0]):
                j = self.shuffle_rng.randint(m)
                # Copy ensures that memory is not aliased.
                tmp = topo_view[i, :, :, :].copy()
                topo_view[i, :, :, :] = topo_view[j, :, :, :]
                topo_view[j, :, :, :] = tmp
        super(CustomMMPosterior, self).__init__(topo_view=dimshuffle(topo_view))

        assert not N.any(N.isnan(self.X))

        if which_set == 'test':
            assert fit_test_preprocessor is None or \
                (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)
