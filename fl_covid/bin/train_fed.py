import os
import sys
import argparse
import json
import random
import warnings
import time

import collections
import tensorflow as tf
print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_v2_behavior()

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import fl_covid.bin  # noqa: F401

    __package__ = "fl_covid.bin"

# Change these to absolute imports if you copy this script outside the fl_covid package.
from fl_covid import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks.eval import Evaluate, Evaluate_separate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator
from ..callbacks import model_checkpoint, reduce_lr
import numpy as np


def makedirs(path):
    """
    Try to create the directory, pass if the directory exists already, fails otherwise.
    Args
        path: The directory to create
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(args, backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False, config=None, model_config=None):
    """ Creates three models (model, training_model, prediction_model).
    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    if model_config is None:
        model_config = dict()

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    channels = None
    if config and 'dimensions' in config:
        channels = int(config['dimensions']['channels'])

    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, channels=channels, model_config=model_config),
                                       weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, channels=channels, model_config=model_config), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal(),
            # 'mask': 'mean_squared_error',
            'mask': losses.mask_focal(),
        },
        optimizer=tf.keras.optimizers.Adam(lr=args.lr, clipnorm=0.001),

    )


    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args, tensorboad_writer=None):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    if args.evaluation and validation_generator:
        evaluation = Evaluate_separate(prediction_model, validation_generator[:-1], tensorboard_writer=tensorboad_writer,
                              weighted_average=args.weighted_average, max_detections=args.max_detections)
        # evaluation_overall = Evaluate(prediction_model, validation_generator[-1], tensorboard_writer=tensorboad_writer,
        #                       weighted_average=args.weighted_average, max_detections=args.max_detections)
        callbacks.append(evaluation)
        # callbacks.append(evaluation_overall)

    # save the model
    # if args.snapshots:
    if args.snapshot_path:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        print('Saving models to path: ', args.snapshot_path)
        checkpoint = model_checkpoint.ModelCheckpoint(
            training_model,
            os.path.join(args.snapshot_path, '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone,
                                                                                                 dataset_type=args.dataset_type)),
            verbose=1,
            save_best_only=args.best_only,
            monitor="mAP",
            mode='max'
        )
        callbacks.append(checkpoint)

    if args.reduce_lr:
        callbacks.append(reduce_lr.ReduceLROnPlateau(
            training_model,
            monitor='mAP',
            factor=0.1,
            patience=1,
            verbose=1,
            mode='max',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        ))

    # callbacks.append(keras.callbacks.EarlyStopping(
    #     monitor='mAP',
    #     min_delta=0,
    #     patience=4,
    #     verbose=1,
    #     mode='max',
    #     restore_best_weights=True
    # ))

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.
    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'config': args.config,
        'image_min_side': args.image_min_side,
        'image_max_side': args.image_max_side,
        'preprocess_image': preprocess_image,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.2, -0.2),
            max_translation=(0.2, 0.2),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.8, 0.8),
            max_scaling=(1.2, 1.2),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    if args.dataset_type == 'csv':
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                **common_args
            )
        else:
            validation_generator = None
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError(
            "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn(
            'Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type_dummy')

    # csv_parser = subparsers.add_parser('csv')
    # csv_parser.add_argument('--classes', default='mapping.csv', help='Path to a CSV file containing class label mapping.')
    # csv_parser.add_argument('--annotations', default='temp.csv', help='Path to CSV file containing annotations for training.')
    # csv_parser.add_argument('--val-annotations', default='temp.csv', help='Path to CSV file containing annotations for validation (optional).')

    # group_csv_parser = subparsers.add_parser('group_csv')
    # group_csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    # group_csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    # group_csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', default=None, help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights', help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights', default=None, help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights', help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone', default='vgg19', help='Backbone model used by retinanet.', type=str)
    parser.add_argument('--batch-size', default=4, help='Size of the batches.', type=int)
    parser.add_argument('--gpu', default='1', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')

    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=100) # used to be 50
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=100) # used to be 10000
    parser.add_argument('--snapshot-path', help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='../snapshots')
    parser.add_argument('--reduce_lr', help='ReduceLROnPlateau', action='store_true')

    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='../logs')
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', default=False, help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', default=True, help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config', default='./fl_covid/anchors4.ini',
                        help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--best_only', help='Only save the best model.', action='store_true')
    parser.add_argument('--max-detections', help='Number of detections to consider during evaluation.', type=int, default=100)
    parser.add_argument('--model-config', help='Path to a model configuration.', type=str)
    parser.add_argument('--dataset_type', default='csv', help='Arguments for specific dataset types')

    parser.add_argument('--classes', default='mapping.csv', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('--annotations', default='temp.csv', help='Path to CSV file containing annotations for training.')
    parser.add_argument('--val-annotations', default=None, help='Path to CSV file containing annotations for validation (optional).')


    return check_args(parser.parse_args(args))


def main(args=None, model_config=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    makedirs(args.snapshot_path + './bin')
    os.system('cp -r ./fl_covid/bin/train_fed.py %s' % (args.snapshot_path + './bin'))
    os.system('cp train.sh %s' % args.snapshot_path)

    # Set empty config
    if model_config is None:
        if args.model_config:
            model_config = json.load(open(args.model_config))[0]
        else:
            model_config = {}

    # seed
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    anno_base_dir = './data/internal'

    args.annotations = os.path.join(anno_base_dir, args.annotations)
    args.classes = './fl_covid/utils/mapping.csv'

    # print('---Server---\ntraining csv {}, validation csv {}'.format(args.annotations, args.val_annotations))

    # tensorboard writer
    makedirs(args.snapshot_path + '/log')
    tensorboard_writer = tf.summary.create_file_writer(args.snapshot_path + '/log')

    data_path = ['internal_1/h5_normalize', 'internal_2/h5_normalize', 'internal_3/h5_normalize']
    train_data_list = ['train_internal_1.csv', 'train_internal_2.csv', 'train_internal_3.csv']
    # test_data_list = ['eval_internal_1_demo.csv', 'eval_internal_2_demo.csv', 'eval_internal_3_demo.csv']

    client_num = 3

    # initialize client dataset and model
    model_clients = []
    train_data_client = []
    test_data_client = []
    step_client = [args.steps, args.steps, args.steps]
    for i in range(client_num):
        args.annotations = os.path.join(anno_base_dir, data_path[i], train_data_list[i])
        # args.val_annotations = os.path.join(anno_base_dir, data_path[i], test_data_list[i])
        # print('---Client{}---\ntraining csv {}, validation csv {}'.format(i, args.annotations, args.val_annotations))
        print('---Client{}---\ntraining csv {}'.format(i, args.annotations))

        train_generator, _ = create_generators(args, backbone.preprocess_image)

        # create the model and dataset
        if args.snapshot is not None:
            print('Loading model, this may take a second...')
            model = models.load_model(args.snapshot, backbone_name=args.backbone)
            training_model = model
            anchor_params = None
            if args.config and 'anchor_parameters' in args.config:
                anchor_params = parse_anchor_parameters(args.config)
            prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
        else:
            weights = args.weights

            print('Creating model, this may take a second...')
            model, training_model, _ = create_models(
                args,
                backbone_retinanet=backbone.retinanet,
                num_classes=train_generator.num_classes(),
                weights=weights,
                multi_gpu=args.multi_gpu,
                freeze_backbone=args.freeze_backbone,
                config=args.config,
                model_config=model_config
            )

        # this lets the generator compute backbone layer shapes using the actual backbone model
        if 'vgg' in args.backbone or 'densenet' in args.backbone:
            train_generator.compute_shapes = make_shapes_callback(model)
            # validation_generator.compute_shapes = train_generator.compute_shapes

        dataset = tf.data.Dataset.from_generator(
            generator=train_generator, output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
            output_shapes=(tf.TensorShape([args.batch_size, 512, 512, 3]), (tf.TensorShape([args.batch_size, 327360, 5]),
                            tf.TensorShape([args.batch_size, 327360, 2]),  tf.TensorShape([args.batch_size, 512, 512, 2]))))

        dataset = dataset.shuffle(100).repeat(args.steps * args.epochs)

        # mapping function to format intput
        def preprocess(dataset):
            def element_fn(element1, element2):
                return collections.OrderedDict([
                    ('x', element1),
                    ('y', element2),
                ])
            return dataset.map(element_fn)

        # make_one_shot_iterator()
        iterator = iter(preprocess(dataset))

        model_clients.append(model)
        train_data_client.append(iterator)
        # test_data_client.append(validation_generator)

    # _, validation_generator = create_generators(args, backbone.preprocess_image)
    # validation_generator.compute_shapes = train_generator.compute_shapes
    # test_data_client.append(validation_generator)

    # create testing model
    server_model, server_training_model,server_prediction_model = create_models(
            args,
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            config=args.config,
            model_config=model_config
        )

    # # create the callbacks
    callbacks = create_callbacks(
        server_model,
        server_training_model,
        server_prediction_model,
        None,
        args,
        tensorboad_writer=tensorboard_writer
    )

    if args.snapshot:
        resume_epoch = int(args.snapshot.rsplit('/', -1)[-1].split('.')[0].rsplit('_', -1)[-1])

    elif args.weights:
        resume_epoch = int(args.weights.rsplit('/', -1)[-1].split('.')[0].rsplit('_', -1)[-1])
    else:
        resume_epoch = 0
    print('Start from epoch:', resume_epoch)

    # prepare callback
    for callback in callbacks:
        callback.on_train_begin()

    # training start
    for epoch in range(resume_epoch, args.epochs):
        epoch_logs = {}
        print('Epoch:{}'.format(epoch), flush=True)
        total_loss = [0.] * client_num
        classification_loss = [0.] * client_num
        regression_loss = [0.] * client_num
        mask_loss= [0.] * client_num
        start = time.perf_counter()
        
        # update each client locally
        for client_index in range(client_num):
            iterator = train_data_client[client_index]
            training_model = model_clients[client_index]
            for step in range(step_client[client_index]):
                sample = iterator.get_next()
                # print('Step:{}'.format(step))
                x = sample['x']
                y = sample['y']

                history = training_model.fit(
                    x=x,
                    y=y,
                    verbose=0,
                )
                total_loss[client_index] += float(history.history['loss'][0])
                classification_loss[client_index] += float(history.history['classification_loss'][0])
                regression_loss[client_index] += float(history.history['regression_loss'][0])
                # mask_loss[client_index] += float(history.history['mask_loss'][0])
                dur = time.perf_counter() - start
                print("\rClient {} Step:[{}/{}] time:{:.2f}s | client average: -total loss:{:.4e} -classfication_loss:{:.4e} -regression_loss:{:.4e}".
                      format(client_index, (step+1), step_client[client_index], dur, total_loss[client_index]/(step+1), classification_loss[client_index]/(step+1), regression_loss[client_index]/(step+1)), end="", flush=True)
            print()
            # model aggregation
        parameter_clients = []
        new_parameters = []
        for client_index in range(client_num):
            parameter_clients.append(model_clients[client_index].get_weights())

        client_weight = [0.15, 0.10, 0.75]
        for param_i in range(len(parameter_clients[0])):
            new_parameter = 0
            for client_i in range(client_num):
                new_parameter += parameter_clients[client_i][param_i] * client_weight[client_i]
            new_parameters.append(new_parameter)

        # re-distribute
        for client_index in range(len(model_clients)):
            model_clients[client_index].set_weights(new_parameters)
        # update testing model 
        server_training_model.set_weights(new_parameters)
        server_model.set_weights(new_parameters)

        # epoch end
        if tensorboard_writer is not None:
            with tensorboard_writer.as_default():
                for client_idx in range(client_num):
                    tf.summary.scalar("loss{}".format(client_idx), total_loss[client_idx]/args.steps, step=epoch)
                    tf.summary.scalar("classification_loss{}".format(client_idx), classification_loss[client_idx]/args.steps, step=epoch)
                    tf.summary.scalar("regression_loss{}".format(client_idx), regression_loss[client_idx]/args.steps, step=epoch)
                    tf.summary.scalar("mask_loss{}".format(client_idx), mask_loss[client_idx]/args.steps, step=epoch)
                # average loss
                tf.summary.scalar("loss", sum(total_loss) / client_num / args.steps, step=epoch)
                tf.summary.scalar("classification_loss", sum(classification_loss) / client_num / args.steps, step=epoch)
                tf.summary.scalar("regression_loss", sum(regression_loss) / client_num / args.steps, step=epoch)
                tf.summary.scalar("mask_loss", sum(mask_loss) / client_num / args.steps, step=epoch)
                tensorboard_writer.flush()

        for callback in callbacks:
            epoch_logs = callback.on_epoch_end(epoch, epoch_logs)

    print('Finish training')
    return training_model, prediction_model


if __name__ == '__main__':
    main()
