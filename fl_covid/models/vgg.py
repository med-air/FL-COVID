import tensorflow.keras as keras
from keras.utils import get_file
from . import Backbone
from ..utils.image import preprocess_image


class VGGBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return vgg_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        if self.backbone == 'vgg16':
            resource = keras.applications.vgg16.vgg16.WEIGHTS_PATH_NO_TOP
            checksum = '6d6bbae143d832006294945121d1f1fc'
        elif self.backbone == 'vgg19':
            resource = keras.applications.vgg19.vgg19.WEIGHTS_PATH_NO_TOP
            checksum = '253f8cb515780f3b799900260a226db6'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg16', 'vgg19', 'vgg19_nl']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def vgg_retinanet(num_classes, backbone='vgg16', inputs=None, modifier=None, channels=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        if channels is None:
            channels = 3

        inputs = keras.layers.Input(shape=(None, None, channels))

    # create the vgg backbone
    if backbone == 'vgg16':
        vgg = keras.applications.VGG16(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'vgg19' or backbone == 'vgg19_nl':
        vgg = keras.applications.VGG19(input_tensor=inputs, include_top=False, weights=None)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model
    layer_names = ["block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    layer_outputs = [inputs] + [vgg.get_layer(name).output for name in layer_names]
    if backbone == 'vgg19':
        from . import retinanet
        return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
    elif backbone == 'vgg19_nl':
        from . import retinanet_nl
        return retinanet_nl.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
    else:
        raise ValueError('Unknown backbone and RetinaNet type \n '
                         '(allowed here are \'vgg19\' or \'vgg19_nl\' or \'vgg19_nl_P3_P4\' or \'vgg19_nl_P5_P6\')')
