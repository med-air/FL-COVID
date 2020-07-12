import tensorflow.keras as keras
import tensorflow as tf
from . import assert_training_model
from .. import initializers
from .. import layers
from .non_local import non_local_block
from ..utils.anchors import AnchorParameters


def default_classification_model(num_classes, num_anchors, model_config, pyramid_feature_size=256, prior_probability=0.01, classification_feature_size=256, name='classification_submodel'):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'strides': 1,
        'padding': 'same',
    }

    submodel_depth = model_config['submodel_depth'] if 'submodel_depth' in model_config else 4
    submode_kernel_size = model_config['submode_kernel_size'] if 'submode_kernel_size' in model_config else 3

    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(submodel_depth):
        outputs = keras.layers.Conv2D(
            kernel_size=submode_kernel_size,
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        kernel_size=submode_kernel_size,
        filters=num_classes * num_anchors,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values, num_anchors, model_config, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    submodel_depth = model_config['submodel_depth'] if 'submodel_depth' in model_config else 4
    submode_kernel_size = model_config['submode_kernel_size'] if 'submode_kernel_size' in model_config else 3

    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(submodel_depth):
        outputs = keras.layers.Conv2D(
            kernel_size=submode_kernel_size,
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, kernel_size=submode_kernel_size, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def AttentionBlock(x, g, name):
    g1 = keras.layers.BatchNormalization(name=name+'_g1BN')(g)
    x1 = keras.layers.BatchNormalization(name=name+'_x1BN')(x)

    g1_x1 = keras.layers.Add()([g1, x1])
    psi = keras.layers.Activation('relu')(g1_x1)
    psi_Conv = keras.layers.Conv2D(1, kernel_size=1,name=name+'_psiConv')(psi)
    psi_BN = keras.layers.BatchNormalization(name=name+'_psiBN')(psi_Conv)
    psi = keras.layers.Activation('sigmoid')(psi_BN)
    x = keras.layers.Multiply()([x, psi])
    return x, psi


def __create_pyramid_features(C0, C1, C2, C3, C4, C5, model_config, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    top_layer = model_config['top_layer'] if 'top_layer' in model_config else 2
    top_connection = model_config['top_connection'] if 'top_connection' in model_config else 2
    top_connection_kernel = model_config['top_connection_kernel'] if 'top_connection_kernel' in model_config else 3
    connection_kernel = model_config['connection_kernel'] if 'connection_kernel' in model_config else 1
    merge_function = keras.layers.Concatenate if 'merge_function' in model_config and model_config['merge_function'] == 1 else keras.layers.Add
    fpn_layers = model_config['fpn_layers'] if 'fpn_layers' in model_config else 1

    P5 = keras.layers.Conv2D(feature_size, kernel_size=connection_kernel, strides=1, padding='same', name='C5_reduced')(C5)
    P5 = non_local_block(P5,out_channels=feature_size,height=16,width=16, compression=1,mode='gaussian', name='C5_nonlocal')
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    for i in range(fpn_layers):
        name = f'P5_{i+1}' if i != fpn_layers - 1 else 'P5'
        P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name)(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=connection_kernel, strides=1, padding='same', name='C4_reduced')(C4)
    P4_y = non_local_block(P4,out_channels=feature_size,height=32,width=32, compression=1, mode='dot', add_residual=False, name='C4_nonlocal')
    P4_y, _ = AttentionBlock(P4_y, P5_upsampled,name='P4Attn')
    P4 = keras.layers.add([P4, P4_y])

    P4 = merge_function(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    for i in range(fpn_layers):
        name = f'P4_{i+1}' if i != fpn_layers - 1 else 'P4'
        P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name)(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=connection_kernel, strides=1, padding='same', name='C3_reduced')(C3)
    P3_y = non_local_block(P3, out_channels=feature_size, height=64, width=64, compression=1, mode='dot', add_residual=False, name='C3_nonlocal')
    P3_y, _ = AttentionBlock(P3_y, P4_upsampled,name='P3Attn')
    P3 = keras.layers.add([P3,P3_y])
    P3 = merge_function(name='P3_merged')([P4_upsampled, P3])
    P3_upsampled = layers.UpsampleLike(name='P3_upsampled')([P3, C2])

    for i in range(fpn_layers):
        name = f'P3_{i+1}' if i != fpn_layers - 1 else 'P3'
        P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name)(P3)

    # add P3 elementwise to C2
    P2 = keras.layers.Conv2D(feature_size, kernel_size=connection_kernel, strides=1, padding='same', name='C2_reduced')(C2)
    P2, _ = AttentionBlock(P2, P3_upsampled,name='P2Attn')
    P2 = merge_function(name='P2_merged')([P3_upsampled, P2])

    for i in range(fpn_layers):
        name = f'P2_{i+1}' if i != fpn_layers - 1 else 'P2'
        P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name)(P2)

    # Pixel wise segmentation
    P2_upsampled = layers.UpsampleLike(name='P2_upsampled')([P2, C1])

    for i in range(top_layer):
        P2_upsampled = keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', name=f'P2_upsampled_conv{i+1}')(P2_upsampled)

    P1 = keras.layers.Conv2D(64, kernel_size=connection_kernel, strides=1, padding='same', activation='relu', name='C1_reduced')(C1)
    P1, _ = AttentionBlock(P1, P2_upsampled,name='P1Attn')
    P1 = merge_function(name='P1_merged')([P2_upsampled, P1])
    P0 = C0

    for i in range(top_connection):
        P0 = keras.layers.Conv2D(32, kernel_size=top_connection_kernel, strides=1, padding='same', activation='relu', name=f'P0_conv{i+1}')(C0)

    P1_upsampled = layers.UpsampleLike(name='P1_upsampled')([P1, P0])

    for i in range(top_layer):
        P1_upsampled = keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', name=f'P1_upsampled_conv{i+1}')(P1_upsampled)

    P0, attention_map = AttentionBlock(P0, P1_upsampled,name='P0Attn')
    P0 = merge_function(name='P0_merged')([P1_upsampled, P0])
    P0 = keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='same', name='P0',
                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                             bias_initializer=initializers.PriorProbability(probability=0.01))(P0)
    mask = keras.layers.Activation('sigmoid', name='mask')(P0)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)
    P6 = non_local_block(P6, out_channels=feature_size, height=8,width=8, compression=1, mode='dot', name='P6_nonlocal')
    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    # P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    # P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return mask, attention_map, [P2, P3, P4, P5, P6]


def default_submodels(num_classes, num_anchors, model_config):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(4, num_anchors, model_config)),
        ('classification', default_classification_model(num_classes, num_anchors, model_config))
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
        inputs,
        backbone_layers,
        num_classes,
        num_anchors=None,
        create_pyramid_features=__create_pyramid_features,
        submodels=None,
        name='retinanet',
        model_config=None):
    """ Construct a RetinaNet model on top of a backbone.
    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [regression, classification, other[0], other[1], ...]
        ```
    """
    print("#### Using RetinaNet with non-local blocks on (P3 P4 P5 P6) ####")
    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors, model_config)

    C0, C1, C2, C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    mask, attention_map, features = create_pyramid_features(C0, C1, C2, C3, C4, C5, model_config)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    outputs = pyramids + [mask] + [attention_map]
    # return keras.models.Model(inputs=inputs, outputs=(pyramids[0], pyramids[1], mask), name=name)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def retinanet_bbox(
        model=None,
        nms=True,
        class_specific_filter=True,
        score_threshold=0.05,
        name='retinanet-bbox',
        anchor_params=None,
        **kwargs):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.
    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        anchor_params         : Struct containing anchor parameters. If None, default values are used.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [boxes, scores, labels, other[0], other[1], ...]
        ```
    """

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P2', 'P3', 'P4', 'P5', 'P6']]
    anchors = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])
    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections',
        score_threshold=score_threshold
    )([boxes, classification])

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections + other, name=name)
