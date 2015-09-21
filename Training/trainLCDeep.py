
import os

from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
import time
from os.path import basename, splitext


@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train():

    start_time = time.time()

    nvis = 1388

    hyper_params = {"hidden_layer_dim" : round(nvis*1.5),
		    "monitoring_batches" : 20,
            "batch_size" : 100,
            "model1" : os.environ['MMDAErbms'] + 'laser_best.pkl',
            "data1" : os.environ['MMDAEdata'] + 'short_laser.npy',
            "model2" : os.environ['MMDAErbms'] + 'command_best.pkl',
            "data2" : os.environ['MMDAEdata'] + 'short_command.npy',
            "type" : 'laser_command',
		    "save_path": os.environ['MMDAErbms'],
            "nvis": nvis}
    yaml = open(os.environ['MMDAEyaml'] + 'deep.yaml', 'r').read()
    yaml = yaml %(hyper_params)
    train_yaml(yaml)
    print (time.time() - start_time)

if __name__ == '__main__':
    train()
