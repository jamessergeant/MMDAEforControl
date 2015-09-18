
import os

from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
import time


@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train():

    sequence = [5]
    start_time = time.time()

    for i in sequence:
        hyper_params = {"hidden_layer_dim" : round(i*181*1.5),
    		    "monitoring_batches" : 20,
                "batch_size" : 100,
                "dataset" : '/home/james/Dropbox/University/BEB801/datasets/straight_laser.npy',
                "type" : 'laser',
    		    "save_path": '/home/james/test/trained_models',
                "sequence": i,
                "nvis": 181*i,
                "init_lr": 0.001,
                "normalise": 1}
        yaml = open(os.environ['MMDAEyaml'] + 'rbm_first.yaml', 'r').read()
        yaml = yaml %(hyper_params)
        train_yaml(yaml)
        print (time.time() - start_time)

if __name__ == '__main__':
    train()
