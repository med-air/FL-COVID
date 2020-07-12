from __future__ import print_function

import keras
import sys

minimum_keras_version = 2, 2, 4


def keras_version():
    """ Get the Keras version.

    Returns
        tuple of (major, minor, patch).
    """
    return tuple(map(int, keras.__version__.split('.')))


def keras_version_ok():
    """ Check if the current Keras version is higher than the minimum version.
    """
    return keras_version() >= minimum_keras_version


def assert_keras_version():
    """ Assert that the Keras version is up to date.
    """
    detected = keras.__version__
    required = '.'.join(map(str, minimum_keras_version))
    assert(keras_version() >= minimum_keras_version), 'You are using keras version {}. The minimum required version is {}.'.format(detected, required)


def check_keras_version():
    """ Check that the Keras version is up to date. If it isn't, print an error message and exit the script.
    """
    try:
        assert_keras_version()
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
