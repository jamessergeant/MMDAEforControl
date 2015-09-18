import os
from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
import time
from pylearn2.utils import serial
from pylearn2.models.mlp import FlattenerLayer, CompositeLayer, Sigmoid, LinearGaussian
from customAE import SplitterLayer
import numpy as N

@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    
    laser_model = serial.load('/home/n8382921/bin/BEB801/trained_models/straight_seq_valid/straight_1358_seq5_valid_best.pkl')
    command_model = serial.load('/home/n8382921/bin/BEB801/trained_models/command/straight_command_seq/straight_command_seq_5_30_best.pkl')
    deep_model = serial.load('/home/n8382921/bin/BEB801/trained_models/multimodal/straight/straight_straight_1358_seq5_valid_best_straight_command_seq_5_30_best_best.pkl')
    
    models = [laser_model,deep_model,deep_model,laser_model,command_model]

    layers = list()

    if os.getenv('THEANO_FLAGS') is not None and 'float32' in os.getenv('THEANO_FLAGS'):
        f = 'float32'
    else:
        f = 'float64'
    
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
                    layer.set_weights(model.get_weights().transpose().astype(f)) # CAN'T USE TIED WEIGHTS ON SOME LAYERS
                layer.set_biases(model.get_param_values()[0].astype(f))
            elif type(layer) is LinearGaussian:
                params = model.get_param_values()
                if layer.enc_layer is None:
                    layer.set_weights(params[2].transpose().astype(f)) # CAN'T USE TIED WEIGHTS ON SOME LAYERS
                layer.set_biases(params[1].astype(f))
                layer.beta.set_value(model.get_params()[0].eval().astype(f))

    del models
    del laser_model
    del command_model
    del deep_model

    train.main_loop()

def train():

    start_time = time.time()

    sequence = 5

    command_multiplier = 2

    overcomplete_multiplier = 1.5

    laser_dim = 181

    command_dim = 2

    hyper_params = {"laser_dataset" : '/home/n8382921/bin/BEB801/datasets/straight_short_laser_aligned.npy',
        "laser_model_base" : 'straight_1358_seq5_valid_best',
        "command_model_base" : 'straight_command_seq_5_30_best',
        "command_dataset" : '/home/n8382921/bin/BEB801/datasets/straight_short_cmdvel_aligned.npy',
        "sequence" : sequence,
        "batch_size" : 100,
        "nvis_laser" : sequence*laser_dim,
        "nvis_command" : sequence*command_dim,
        "h11_dim" : round(sequence*laser_dim*overcomplete_multiplier),
        "h12_dim" : round(sequence*command_multiplier*command_dim*overcomplete_multiplier),
        "h2_dim" : round((round(sequence*laser_dim*overcomplete_multiplier) + round(sequence*command_multiplier*command_dim*overcomplete_multiplier)) * overcomplete_multiplier),
        "h3_dim": round(sequence*laser_dim*overcomplete_multiplier) + round(sequence*command_multiplier*command_dim*overcomplete_multiplier),
        "type" : 'mm_short_single',
        "save_path": '/home/n8382921/bin/BEB801/trained_models'}
    yaml = open("/home/n8382921/bin/BEB801/networks/MM_MLP/mm_short_single.yaml", 'r').read()
    yaml = yaml %(hyper_params)
    train_yaml(yaml)
    print (time.time() - start_time)

if __name__ == '__main__':
    train()
