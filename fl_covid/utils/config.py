import configparser
import numpy as np
import tensorflow.keras as keras
from ..utils.anchors import AnchorParameters


def read_config_file(config_path):
    config = configparser.ConfigParser()

    with open(config_path, 'r') as file:
        config.read_file(file)

    assert 'anchor_parameters' in config, \
        "Malformed config file. Verify that it contains the anchor_parameters section."

    assert set(config['anchor_parameters']) <= set(AnchorParameters.default.__dict__.keys()), \
        "Malformed config file. Verify that there are no typos in the keys."

    return config


def parse_anchor_parameters(config):
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

    return AnchorParameters(sizes, strides, ratios, scales)
