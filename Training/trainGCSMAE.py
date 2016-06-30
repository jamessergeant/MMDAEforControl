import os
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
import time
from pylearn2.utils import serial
from pylearn2.models.mlp import FlattenerLayer, CompositeLayer, Sigmoid, LinearGaussian
from custom import SplitterLayer
import numpy as N
import theano

@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)

    mode1 = serial.load(os.environ['MMDAErbms'] + '/goal_best.pkl')
    mode2 = serial.load(os.environ['MMDAErbms'] + '/command_best.pkl')
    deep = serial.load(os.environ['MMDAErbms'] + '/goal_command_best.pkl')

    models = [mode1,deep,deep,mode1,mode2]

    layers = list()

    f = theano.config.floatX

    for ii, layer in enumerate(train.model.layers):
        if type(layer) is FlattenerLayer:
            for l in layer.raw_layer.layers:
                layers.append(l)
        elif type(layer) is SplitterLayer:
            layers.append(layer.raw_layer)
        else:
            layers.append(layer)

    for ii, (layer,model) in enumerate(zip(layers,models)):
        if ii < len(layers)/2:
            if type(layer) is Sigmoid:
                if layer.get_weights().shape != model.get_weights():
                    layer.set_weights(model.get_weights()[:layer.get_weights().shape[0],:].astype(f))
                else:
                    layer.set_weights(model.get_weights().astype(f))
                if len(model.get_param_values()) == 4:
                    layer.set_biases(model.get_param_values()[3].astype(f))
                else:
                    layer.set_biases(model.get_param_values()[2].astype(f))
        else:
            if type(layer) is Sigmoid:
                if layer.enc_layer is None:
                    layer.set_weights(model.get_weights().transpose().astype(f))
                layer.set_biases(model.get_param_values()[0].astype(f))
            elif type(layer) is LinearGaussian:
                params = model.get_param_values()
                if layer.enc_layer is None:
                    layer.set_weights(params[2].transpose().astype(f))
                layer.set_biases(params[1].astype(f))
                beta = model.get_params()[0].eval()
                if isinstance(beta, N.ndarray):
                    layer.beta.set_value(model.get_params()[0].eval().astype(f))
                elif isinstance(beta, theano.sandbox.cuda.type.CudaNdarrayType):
                    layer.beta.set_value(model.get_params()[0].eval().dtype(f))

    del models
    del mode1
    del mode2
    del deep

    train.main_loop()

def train():

    start_time = time.time()

    sequence = 5

    mode1_multiplier = 2
    mode2_multiplier = 2

    overcomplete_multiplier = 1.5

    mode1_dim = 2

    mode2_dim = 2

    hyper_params = {"mode1_dataset" : os.environ['MMDAEdata'] + '/open_goal.npy',
        "mode2_dataset" : os.environ['MMDAEdata'] + '/open_command.npy',
        "sequence" : sequence,
        "batch_size" : 100,
        "nvis_mode1" : sequence*mode1_dim,
        "nvis_mode2" : sequence*mode2_dim,
        "h11_dim" : round(sequence*mode1_multiplier*mode1_dim*overcomplete_multiplier),
        "h12_dim" : round(sequence*mode2_multiplier*mode2_dim*overcomplete_multiplier),
        "h2_dim" : round((round(sequence*mode1_multiplier*mode1_dim*overcomplete_multiplier) + round(sequence*mode2_multiplier*mode2_dim*overcomplete_multiplier)) * overcomplete_multiplier),
        "h3_dim": round(sequence*mode1_multiplier*mode1_dim*overcomplete_multiplier) + round(sequence*mode2_multiplier*mode2_dim*overcomplete_multiplier),
        "type" : 'goal_command_smae',
        "save_path": os.environ['MMDAEdaes']}
    yaml = open(os.environ['MMDAEyaml'] + '/smae.yaml', 'r').read()
    yaml = yaml %(hyper_params)
    train_yaml(yaml)
    print (time.time() - start_time)

if __name__ == '__main__':
    train()
