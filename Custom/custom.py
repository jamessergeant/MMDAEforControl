from pylearn2.models.mlp import Layer
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import wraps
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.utils import serial
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

import theano
from theano.compat.six.moves import xrange
from theano import tensor as T

import numpy as N
np = N

import os
import os.path
from os.path import basename, splitext

class SplitterLayer(Layer):

    def __init__(self, raw_layer, split):
        super(SplitterLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.layer_name = raw_layer.layer_name
        self.split = split

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.raw_layer.set_input_space(space)
        self.output_space = CompositeSpace((VectorSpace(self.split[0]),VectorSpace(self.split[1])))

    @wraps(Layer.get_input_space)
    def get_input_space(self):
        return self.raw_layer.get_input_space()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        return self.raw_layer.get_monitoring_channels(data)

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        raw_space = self.raw_layer.get_output_space()
        state = raw_space.undo_format_as(state,
                                         self.get_output_space())

        if targets is not None:
            targets = self.get_target_space().format_as(
                targets, self.raw_layer.get_target_space())
        return self.raw_layer.get_layer_monitoring_channels(
            state_below=state_below,
            state=state,
            targets=targets
        )

    @wraps(Layer.get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        return self.raw_layer.get_monitoring_data_specs()

    @wraps(Layer.get_params)
    def get_params(self):
        return self.raw_layer.get_params()

    @wraps(Layer.get_weights)
    def get_weights(self):
        return self.raw_layer.get_weights()

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return self.raw_layer.get_weight_decay(coeffs)

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return self.raw_layer.get_l1_weight_decay(coeffs)

    @wraps(Layer.set_batch_size)
    def set_batch_size(self, batch_size):
        self.raw_layer.set_batch_size(batch_size)

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        self.raw_layer.modify_updates(updates)

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        return self.raw_layer.get_lr_scalers()

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        raw = self.raw_layer.fprop(state_below)

        return self.raw_layer.get_output_space().format_as(raw,
                                                           self.output_space)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        raw_space = self.raw_layer.get_output_space()
        target_space = self.output_space

        raw_Y = target_space.format_as(Y, raw_space)
        raw_Y_hat = raw_space.undo_format_as(Y_hat, target_space)
        raw_space.validate(raw_Y_hat)

        return self.raw_layer.cost(raw_Y, raw_Y_hat)

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):

        super(SplitterLayer, self).set_mlp(mlp)
        self.raw_layer.set_mlp(mlp)

class monReconErr(DefaultDataSpecsMixin, Cost):
    def __init__(self):

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        recon = model.reconstruct(data)
        return (N.multiply(recon - data,recon - data)).mean()

class monKurtosis(DefaultDataSpecsMixin, Cost):
    def __init__(self):

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data):

        weights = model.get_weights()
        error = weights - weights.mean()
        kurtosis = weights.shape[0]*weights.shape[1]*N.sum(N.power(error,4)) / N.power(N.sum(N.power(error,2)),2) - 3
        return T.TensorVariable(kurtosis)

class CustomMMLoaderDropout(VectorSpacesDataset):

    def __init__(self,datasets,which_set,sequence=1,dropout=True,normalise=[],labels=[],shuffle=False,start=None,stop=None):

        self.args = locals()

        assert which_set in ['train','valid']
        assert len(normalise) == len(datasets)

        vector_spaces = tuple()
        data = tuple()
        self.mean = list()
        self.std = list()
        z = list()
        for dataset, n in zip(datasets,normalise):
            if dataset[-3] == 'npz':
                mode_data = N.load(dataset)['arr_0']
            else:
                mode_data = N.load(dataset)

            z.append((mode_data == 0).all(axis=1))

            r,c = mode_data.shape

            if start is not None:
                assert stop is not None
                assert start >= 0
                assert stop > start
                if stop > mode_data.shape[0]:
                    raise ValueError('stop=' + str(stop) + '>' +
                                     'm=' + str(mode_data.shape[0]))
                mode_data = mode_data[start:stop, :]
                if mode_data.shape[0] != stop - start:
                    raise ValueError("data.shape[0]: %d. start: %d stop: %d"
                                     % (mode_data.shape[0], start, stop))

            cuts = (mode_data == 0).all(axis=1).nonzero()
            mean = 1
            std = 1

            if n == 1:
                mean = mode_data.mean()
                std = mode_data.std()
            elif n ==2:
                mean = mode_data.mean(axis=0)
                std = mode_data.std(axis=0)

            mode_data = (mode_data - mean) / std
            self.mean.append(mean)
            self.std.append(std)

            if sequence != 1:
                temp = mode_data
                mode_data = np.zeros([temp.shape[0],sequence*temp.shape[1]])
                for i in range(0,temp.shape[0]-sequence):
                    mode_data[i,:] = temp[i:i+sequence,:].reshape(sequence*temp.shape[1])
                del temp

            vector_spaces = vector_spaces + (VectorSpace(mode_data.shape[1]),)
            data = data + (mode_data,)

        # # remove changes between datasets if stacked
        # b = N.logical_and(z[0],z[1])
        # for a in z[2:]:
        #     b = N.logical_and(b,a)
        # for ii in range(b.shape[0]):
        #     if b[ii]:
        #         for i in range(ii-sequence+1,ii):
        #             b[i] = True
        # ind = N.logical_not(b).nonzero()
        # for ii in range(len(data)):
        #     data[ii] = data[ii][ind,:]
        #
        # assert data[0].shape[0] == data[1].shape[0]
        # assert data[2].shape[0] == data[1].shape[0]
        # assert data[2].shape[0] == data[0].shape[0]

        ground_truth = data[0].copy()
        for ii in data[1:]:
            ground_truth = N.concatenate((ground_truth,ii),axis=1)

        # Modal Dropout - drop only 1 mode at a time
        if dropout:

            data = list(data)
            seq = N.ones((1,3))
            for ii in range(len(datasets)):
                seq2 = seq.copy()
                seq2[:,ii] = N.zeros((seq.shape[0]))
                seq = N.concatenate((seq,seq2),axis=0)
            seq = seq[1:-1,:]

            cases = N.where(seq.sum(axis=1) != len(datasets)-1)
            seq = N.delete(seq,cases,axis=0)

            all_data = list(data)
            all_data.append(ground_truth)

            for ii in all_data:
                assert not N.isnan(N.sum(ii))
            for i,ii in enumerate(seq):
                for j,jj in enumerate(ii):
                    if jj:
                        all_data[j] = N.concatenate((all_data[j],data[j]),axis=0)
                    else:
                        all_data[j] = N.concatenate((all_data[j],N.zeros(data[j].shape)),axis=0)
                all_data[-1] = N.concatenate((all_data[-1],ground_truth),axis=0)

            data = tuple(all_data)
            del all_data
        else:
            data = list(data)
            data.append(ground_truth)
            data = tuple(data)
        vector_spaces = vector_spaces + (VectorSpace(data[-1].shape[1]),)

        if shuffle:
            self.shuffle_rng = make_np_rng(
                None, [1, 2, 3], which_method="shuffle")
            for ii in xrange(data[0].shape[0]):
                jj = self.shuffle_rng.randint(data[0].shape[0])
                # Copy ensures that memory is not aliased.
                for d in data:
                    tmp = d[ii,:].copy()
                    d[ii,:] = d[jj,:]
                    d[jj,:] = tmp

        if len(labels) == 0:
            for ii in range(len(datasets)):
                labels.append('dataset_%i' % ii)
        labels.append('targets')
        data_specs = (CompositeSpace(vector_spaces), tuple(labels))
        super(CustomMMLoaderDropout,self).__init__(data=data,data_specs=data_specs)


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

        super(CustomLoader, self).__init__(topo_view=dimshuffle(topo_view))

        assert not N.any(N.isnan(self.X))

# For use in greedy layer-wise training of multi-modal deep layer RBM
class CustomMMPosterior(DenseDesignMatrix):

    def __init__(self, models, datasets, normalise, which_set, batch_size, shuffle=False,
                 start=None, stop=None, length=None,
                 axes=['b', 0, 1, 'c']):

        self.args = locals()
        existing_data_path = os.environ['MMDAEdata'] + splitext(basename(models[0]))[0] + '_' + splitext(basename(datasets[0]))[0] + '_' + splitext(basename(models[1]))[0] + '_' + splitext(basename(datasets[1]))[0] + '_' + which_set + '.npy'
        assert which_set in ['train', 'valid']
        assert type(models) is list
        assert type(datasets) is list
        assert len(models) == len(datasets)
        assert len(models) == len(normalise)

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        # only process if it hasn't been done already
        if not os.path.exists(existing_data_path):
            for ii in range(len(models)):
                # Load single mode model for first layer
                model = serial.load(models[ii])
                # Load the single mode data
                data = N.load(datasets[ii])

                if normalise[ii] == 1:
                    data_mean = data.mean()
                    data_std = data.std()
                elif normalise[ii] == 2:
                    data_mean = data.mean(axis=0)
                    data_std = data.std(axis=0)

                data = (data-data_mean) / data_std

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
