#!/usr/bin/env python
import argparse
import os
import sys

import csv
import h5py
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import SimpleITK as sitk
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_v2_behavior()


if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import fl_covid.bin  # noqa: F401
    __package__ = "fl_covid.bin"

# Change these to absolute imports if you copy this script outside the fl_covid package.
from ..utils.anchors import compute_overlap
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.eval import _compute_ap, _get_annotations, _get_annotations_and_img_path
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.visualization import draw_detections, draw_annotations
from keras_retinanet.utils.visualization import draw_box, label_color, draw_caption
from keras_retinanet.bin.train_edit import create_models
from keras_retinanet.layers.filter_detections import filter_detections
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.bin.evaluate_internal_patient_wise import draw_colorful_result, evaluate_from_npy

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)



def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, slice_id=None, bbox_writer=None, score_threshold=0.4):  # score_threshold used to be 0.5
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]
    for i in selection:
        c = color if color is not None else label_color(labels[i])

        if bbox_writer is not None and slice_id is not None:
            tar_path = 'slice_{}.png'.format(slice_id)
            b = np.array(boxes[i, :]).astype(int)
            bbox_writer.writerow([tar_path]+ [b[0],b[1],b[2],b[3]]+['lesion'])

        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else str(labels[i])) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)




def create_generator(args):
    """ Create generators for evaluation.
    """

    if args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator

def _seg_filter(bboxes,scores_sort,seg):
    # print('scores_sort')
    # print(scores_sort.shape)
    # print(scores_sort)
    # print('indices')
    # print(type(indices[scores_sort]))
    # print(indices[scores_sort])
    # print(indices)
    # # select detections
    # image_boxes      = boxes[0, indices[scores_sort], :]
    # print('seletec_boxes', image_boxes.shape)
    # print(type(image_boxes))
    # print(image_boxes)
    # image_boxes_filtered = []
    # seg = cv2.imread('/Users/jemary/Data/DataSet/COVID-19/COVID-19 image data collection(public1)/007_seg/slice_seg_{}.png'.format(img_idx))
    # print(seg.shape)
    image_boxes = bboxes
    inner = np.asarray([],dtype=np.bool)
    flag = False
    for i in range(image_boxes.shape[0]):
        x1 = int(image_boxes[i][0])
        y1 = int(image_boxes[i][1])
        x2 = int(image_boxes[i][2])
        y2 = int(image_boxes[i][3])
        x1 = 511 if x1 > 511 else x1
        y1 = 511 if y1 > 511 else y1
        x2 = 511 if x2 > 511 else x2
        y2 = 511 if y2 > 511 else y2
        # print(scores_sort)
        # print(scores_sort.shape)
        if (seg[y1,x1,:] == 0).all() and (seg[y2,x2,:] == 0).all() and  (seg[y1,x2,:] == 0).all() and  (seg[y2,x1,:] == 0).all():
            inner = np.append(inner,False)
            flag=True
            # scores_sort = np.delete(scores_sort,i,axis=0)
        else:
            inner = np.append(inner, True)
    # print(inner)
    # cnt = 1
    # if flag:
    #     if cnt > 0:
    #         print("FP out of lung filtered")
    #     cnt -= 1
    scores_sort = scores_sort[inner]
    # print('scores_sort after filter')
    # print(scores_sort.shape)
    # print(scores_sort)
    return scores_sort



def _print_ensemble_detections_to_npy(args, generator, model_list, client_idx, client_name, patient_name, score_threshold=0.05, max_detections=100, save_path=None):
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    detection_out = np.zeros([generator.size(),512,512,3])
    # detection_out = np.zeros([generator.size(),512,512])
    attention_out = np.zeros([generator.size(),512,512])
    mask_out = np.zeros([generator.size(),512,512])

    results = open(os.path.join(save_path, '{}_{}_output_bbox.csv'.format(client_name, patient_name)), 'w', newline='')
    result_writer = csv.writer(results, delimiter=',')

    for i in tqdm(range(generator.size()), desc='Running network on {} {}: '.format(client_name, patient_name)):
        raw_image    = generator.load_image(i)
        # image = np.expand_dims(raw_image.copy(), axis=-1)
        # image = np.repeat(image, 3, axis=-1)
        # image        = generator.preprocess_image(image)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        # boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        all_boxes = np.empty([1,0,4],dtype=np.float32)
        all_scores = np.empty([1,0], dtype=np.float32)
        all_labels = np.empty([1,0], dtype=np.int32)
        masks = np.zeros([1,512,512,1], dtype=np.float32)
        attention_map = np.zeros([1,512,512,1], dtype=np.float32)
        for site_model in model_list:
            site_boxes, site_scores, site_labels, site_masks, site_attention_map = site_model.predict_on_batch(np.expand_dims(image, axis=0))
            site_boxes = site_boxes.numpy()
            site_scores = site_scores.numpy()
            site_labels = site_labels.numpy()
            site_masks = site_masks.numpy()
            site_attention_map = site_attention_map.numpy()
            if np.squeeze(site_attention_map).shape[0] !=512:
                resized_attn_map = cv2.resize(np.squeeze(site_attention_map),(512,512))
                site_attention_map = np.expand_dims(resized_attn_map,axis=-1)
                site_attention_map = np.expand_dims(site_attention_map,axis=0)
            all_boxes = np.concatenate([all_boxes, site_boxes],axis=-2)
            all_scores = np.concatenate([all_scores, site_scores],axis=-1)
            all_labels = np.concatenate([all_labels, site_labels],axis=-1)
            masks = np.add(masks, site_masks/len(model_list))
            attention_map = np.add(attention_map, site_attention_map/len(model_list))
            # print('-------------EACH MODEL----------')
            # print('----boxes----')
            # print(type(site_boxes))
            # print(site_boxes.shape)
            # print(site_boxes.dtype)
            # print('----scores----')
            # print(type(site_scores))
            # print(site_scores.shape)
            # print(site_scores.dtype)
            # print('----labels----')
            # print(type(site_labels))
            # print(site_labels.shape)
            # print(site_labels.dtype)
            # print('----masks----')
            # print(type(site_masks))
            # print(site_masks.shape)
            # print(site_masks.dtype)
            # print('----attn map----')
            # print(type(site_attention_map))
            # print(site_attention_map.shape)
            # print(site_attention_map.dtype)
            # print('total_boxes')
            # print(all_boxes.shape)
            # print(all_boxes.dtype)
            # print('total_scores')
            # print(all_scores.shape)
            # print(all_scores.dtype)
            # print('total_labels')
            # print(all_labels.shape)
            # print(all_labels.dtype)
        # print(np.squeeze(all_boxes).shape)
        # print(np.expand_dims(np.squeeze(all_scores),axis=-1).shape)
        out_boxes, out_scores, out_labels = filter_detections(
            np.squeeze(all_boxes),
            np.expand_dims(np.squeeze(all_scores),axis=-1),
            other=[],
            class_specific_filter=True,
            nms=True)
        out_boxes = np.expand_dims(np.squeeze(out_boxes),axis=0)
        out_scores = np.expand_dims(np.squeeze(out_scores),axis=0)
        out_labels = np.expand_dims(np.squeeze(out_labels),axis=0)
        # print('boxes:', out_boxes.shape)
        # print('scores:', out_scores.shape)
        # print('labels',out_labels.shape)
        boxes = out_boxes.copy()
        scores = out_scores.copy()
        labels = out_labels.copy()

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > -1)[0]
        # print('indices', indices)
        # print(type(scores))
        if type(scores) is not np.ndarray:
            scores = scores.numpy()
            boxes = boxes.numpy()
            labels = labels.numpy()
            masks = masks.numpy()
            attention_map = attention_map.numpy()
        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        image_boxes = boxes[0, indices[scores_sort], :]
        # print('seletec_boxes',image_boxes.shape)
        # print(image_boxes)

        # filter out of lung
        if args.lung_filter:
            client_paths = ['private_1', 'private_2', 'private_3']
            # client_paths = ['private_4/B']
            lung_filter_path = '/research/dept8/qdou/data/covid/{}/lung_seg_png/'.format(client_paths[client_idx])
            # lungfilter = '/covid/private_2/lung_seg_png/
            # print('---img path---')
            img_path = generator.image_path(i)
            patient = img_path.split('/')[-2]
            slice_idx = img_path.split('/')[-1].replace('slice_', '').replace('.h5', '')
            # print('patient:', patient)
            # print('slice:', slice_idx)
            seg_path = os.path.join(lung_filter_path,'{}_slice_{}.png').format(patient,slice_idx)
            # print(seg_path)
            seg = cv2.imread(seg_path)
            scores_sort = _seg_filter(image_boxes,scores_sort,seg)
            image_boxes = boxes[0, indices[scores_sort], :]

        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        if args.save_result == 1:
            img_path = generator.image_path(i)
            img_path = img_path.replace('h5_normalize', 'h5')
            # print(img_path)
            with h5py.File(img_path, "r") as hf:
                h5_raw_image = hf['arr'][:]

            draw_annotations(h5_raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            # draw_detections(raw_image, image_boxes, image_scores, image_labels, score_threshold=args.score_threshold, label_to_name=generator.label_to_name)
            draw_detections(h5_raw_image, image_boxes, image_scores, image_labels, slice_id=i, bbox_writer=result_writer, score_threshold=args.score_threshold)
            # if args.lung_filter:
            # slice_idx = generator.image_path(i).split('/')[-1].replace('slice', '').replace('.png', '')
            # cv2.imwrite('../COVID/slice_{}.png'.format(slice_idx),raw_image)


                # print("Shape of load Image")
                # print(arr.shape)

            detection_out[i, :, :] = h5_raw_image

            attention_map[np.where(attention_map < args.attention_threshold)] = 0
            # attention_out[i, :, :] = cv2.flip( cv2.resize(np.squeeze(np.uint8(attention_map * 255)), (origin_shape[1], origin_shape[0])), 0)
            attention_out[i, :, :] = cv2.resize(np.squeeze(np.uint8(attention_map * 255)), (512, 512))

            masks[masks < args.segmentation_threshold] = 0
            masks = cv2.resize(np.squeeze(np.uint8(masks * 255)), (512, 512))

            mask_out[i, :, :] = masks


    if save_path is not None and args.save_result == 1:
        print('Writing Results...')
        # detection_out = sitk.GetImageFromArray(detection_out)
        # sitk.WriteImage(detection_out, os.path.join(save_path, '{}_{}_detection_result.nii.gz'.format(client_name, patient_name)))

        # attention_out = sitk.GetImageFromArray(attention_out)
        # sitk.WriteImage(attention_out, os.path.join(save_path, '{}_{}_attention_result.nii.gz'.format(client_name, patient_name)))

        mask_out = sitk.GetImageFromArray(mask_out)
        sitk.WriteImage(mask_out, os.path.join(save_path, '{}_{}_masks_result.nii.gz'.format(client_name, patient_name)))

    np.save(os.path.join(save_path, '{}_{}_prediction.npy'.format(client_name, patient_name)), all_detections)
    all_annotations, all_annotations_img_path = _get_annotations_and_img_path(generator)
    np.save(os.path.join(save_path, '{}_{}_annotations.npy'.format(client_name, patient_name)), all_annotations)
    np.save(os.path.join(save_path, '{}_{}_annotations_img_path.npy'.format(client_name, patient_name)), all_annotations_img_path)

    return 0



def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    # subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    # subparsers.required = True

    # csv_parser = subparsers.add_parser('csv')
    # csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    # csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('--ensemble', help='Path to RetinaNet model.', default=False, action='store_true')
    parser.add_argument('--model', help='Path to RetinaNet model.', default=None)
    parser.add_argument('--weights', help='only load weights.', default=None)
    parser.add_argument('--nii',              help='path to nii files.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='vgg19')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.4, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--detection-threshold', help='Threshold used for determining what detections to draw.', default=0.4, type=int)
    parser.add_argument('--segmentation-threshold', help='Threshold used for filter segmentation map.', default=0.1, type=int)
    parser.add_argument('--attention-threshold', help='Threshold used for filter attention map.', default=0.8, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).',default=None)
    parser.add_argument('--get_predicted_bbox',        help='Save predicted bbox to csv.', action='store_true')
    parser.add_argument('--save-result',        help='Save result or not.', type=int, default=0)
    parser.add_argument('--lung-filter',        help='Path for lung seg filter images', default=False, action='store_true')
    parser.add_argument('--draw-colorful', help='draw difficult type of predict with color', default=False, action='store_true')
    parser.add_argument('--reduce-fp', help='reduce fp, must use after completing first evaluation', default=False, action='store_true')
    parser.add_argument('--log',        help='Path for saving log file', default=None)
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=512)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=512)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--dataset_type', help='Path to CSV file containing annotations for evaluation.', default='csv')
    parser.add_argument('--annotations', help='Path to CSV file containing annotations for evaluation.')
    parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.', default='mapping.csv')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)


    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    anno_base_dir = '/research/dept8/qdou/data/covid/'
    args.classes = os.path.join(anno_base_dir, args.classes)

    # create the generator

    # print model summary
    # print(model.summary())

    client_name = ['Dataset1', 'Dataset2', 'Dataset3']
    data_path = ['private_1/h5_normalize', 'private_2/h5_normalize', 'private_3/h5_normalize']
    # data_path = ['private_1/h5_normalize_-1050_800', 'private_2/h5_normalize_-1050_800','private_3/h5_normalize_-1050_800']
    # data_path = ['private_4/B/h5_normalize']
    private_1 = ['P5_annotations_h5_whole_vol.csv']
    private_2 = ['case1_annotations_h5_whole_vol.csv', 'case4_annotations_h5_whole_vol.csv']
    private_3 = ['case19_annotations_h5_whole_vol.csv', 'case23_annotations_h5_whole_vol.csv',
                 'case40_annotations_h5_whole_vol.csv', 'case42_annotations_h5_whole_vol.csv',
                 'case46_annotations_h5_whole_vol.csv', 'case49_annotations_h5_whole_vol.csv',
                 'case51_annotations_h5_whole_vol.csv', 'case54_annotations_h5_whole_vol.csv',
                 'case58_annotations_h5_whole_vol.csv', 'case60_annotations_h5_whole_vol.csv',
                 'case61_annotations_h5_whole_vol.csv', 'case62_annotations_h5_whole_vol.csv']
    private_4 = ['001_annotations_h5_whole_vol.csv', '005_annotations_h5_whole_vol.csv',
                 '006_annotations_h5_whole_vol.csv', '008_annotations_h5_whole_vol.csv',
                 '009_annotations_h5_whole_vol.csv', '010_annotations_h5_whole_vol.csv',
                 '011_annotations_h5_whole_vol.csv', '012_annotations_h5_whole_vol.csv',
                 '013_annotations_h5_whole_vol.csv', '014_annotations_h5_whole_vol.csv']
    # test_data_list = ['test_private_1_all.csv', 'test_mos_all.csv']
    test_data_list = [private_1, private_2, private_3]


    assert len(client_name) == len(data_path) == len(test_data_list)

    # generate patient name based on csv
    patient_names = {}
    for i in range(len(client_name)):
        for j in range(len(test_data_list[i])):
            if client_name[i] not in patient_names:
                patient_names[client_name[i]] = []
                patient_names[client_name[i]].append(test_data_list[i][j].split('_')[0])
            else:
                patient_names[client_name[i]].append(test_data_list[i][j].split('_')[0])



    # start evaluation
    log_path = args.log if args.log else './evaluate_ensemble_internal_patient_wise_3_June.txt'
    logfile = open(log_path,'a')

    # save prediction to npy
    if args.get_predicted_bbox == 1:
        logfile.write('*********************************\n')
        logfile.write('Save prediction of ensemble model to .npy file\n'.format(args.model))

        backbone = models.backbone(args.backbone)

        # optionally load anchor parameters
        anchor_params = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)

        # load model
        if args.ensemble:
            model_list = []
            # model_path_list = ['/research/dept8/qdou/mrjiang/Impriving_retina/final_separate_model/private1/vgg19_nl_csv_15.h5',
            model_path_list = ['/research/dept8/qdou/mrjiang/Impriving_retina/federated_results/21_May_siteB_test_fold2/vgg19_nl_csv_15.h5',
                               '/research/dept8/qdou/mrjiang/Impriving_retina/final_separate_model/private2/vgg19_nl_csv_07.h5',
                               '/research/dept8/qdou/mrjiang/Impriving_retina/final_separate_model/private3/vgg19_nl_csv_17.h5']

            for model_path in model_path_list:
                print('Loading {}...'.format(model_path.split('/')[-2]+model_path.split('/')[-1]),flush=True)
                model = models.load_model(model_path, backbone_name=args.backbone)
                if args.convert_model:
                    model = models.convert_model(model, anchor_params=anchor_params)
                model_list.append(model)
        elif args.model is not None:
            print('Loading model, this may take a second...',flush=True)
            model = models.load_model(args.model, backbone_name=args.backbone)
            # optionally convert the model
            if args.convert_model:
                model = models.convert_model(model, anchor_params=anchor_params)
        elif args.weights is not None:
            weights = args.weights

            print('Creating model and Loading weights, this may take a second...',flush=True)
            model, training_model, prediction_model = create_models(
                backbone_retinanet=backbone.retinanet,
                # note : when mapping.csv only contains lesion,0,  generator.num_classes() ==1
                num_classes=1,
                weights=weights,
                multi_gpu=args.multi_gpu,
                freeze_backbone=False,
                config=args.config,
                model_config={}
            )
            # optionally convert the model
            if args.convert_model:
                model = models.convert_model(model, anchor_params=anchor_params)
        else:
            raise ValueError("You have to specify a model")



        # create generator
        # generators = []
        generators = {}

        for i in range(len(client_name)):
            for j in range(len(test_data_list[i])):
                args.annotations = os.path.join(anno_base_dir, data_path[i], test_data_list[i][j])
                print('---client {}---'.format(client_name[i]))
                print('validation csv {}'.format(args.annotations))
                generator = create_generator(args)
                if client_name[i] not in generators:
                    generators[client_name[i]] = []
                    generators[client_name[i]].append(generator)

                else:
                    generators[client_name[i]].append(generator)


        if args.lung_filter:
            print('do lung filter',flush=True)
            logfile.write('do lung filter\n')
        else:
            print('no lung filter',flush=True)

        for i in range(len(generators)):
            print('------client {}-----'.format(client_name[i]))
            for j in range(len(generators[client_name[i]])):
                logfile.write('Writing client {} patient {} prediction results to .npy... \n'.format(client_name[i], patient_names[client_name[i]]))
                print('------patient {}-----'.format(patient_names[client_name[i]][j]))
                generator = generators[client_name[i]][j]
                patient_name = patient_names[client_name[i]][j]
                if args.ensemble:
                    _print_ensemble_detections_to_npy(
                        args,
                        generator,
                        model_list,
                        client_idx=i,
                        client_name=client_name[i],
                        patient_name=patient_name,
                        score_threshold=args.score_threshold,
                        max_detections=args.max_detections,
                        save_path=args.save_path,
                    )
                else:
                    raise ValueError("This is ensemble code")
        logfile.write('Finish writing \n')
        logfile.write('*********************************')
        sys.exit(0)


    # evaluate from npy
    logfile.write('*********************************\n')
    logfile.write('*{}*\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logfile.write('Evaluate ensemble model from .npy\n'.format(args.model))
    logfile.write('thresshold:{}\n'.format(args.score_threshold))

    for i in range(len(client_name)):
        print('client {}'.format(client_name[i]))
        client_tps, client_fps, client_num_annotations, client_num_slices = 0., 0., 0., 0.
        client_precision, client_recall, client_fp_slice = [], [], []
        client_mAP = []
        for j in range(len(test_data_list[i])):
            patient_name = patient_names[client_name[i]][j]
            average_precisions, old, new, num_annotations, num_slices = evaluate_from_npy(
                args,
                client_name=client_name[i],
                patient_name=patient_name,
                iou_threshold=args.iou_threshold,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                save_path=args.save_path,
            )
            patient_tp = new[0]
            patient_fp = new[1]
            client_recall.append(new[2])
            client_precision.append(new[3])
            client_fp_slice.append(new[4])

            client_tps += patient_tp
            client_fps += patient_fp
            client_num_annotations += num_annotations
            client_num_slices += num_slices

            if args.draw_colorful:
                draw_colorful_result(args,
                                     client_name=client_name[i],
                                     patient_name=patient_name,
                                     iou_threshold=args.iou_threshold,
                                     score_threshold=args.score_threshold,
                                     max_detections=args.max_detections,
                                     save_path=args.save_path)


            # print evaluation
            total_instances = []
            precisions = []
            for label, (average_precision, num_annotations) in average_precisions.items():
                # print('    {:.0f} instances of class'.format(num_annotations),
                #       'lesion', 'with average precision: {:.4f}'.format(average_precision))
                total_instances.append(num_annotations)
                precisions.append(average_precision)

            if sum(total_instances) == 0:
                print('No test instances found.')
                return

            if args.weighted_average:
                print('    mAP: {:.4f}'.format(
                    sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
            else:
                print('    mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

            mAP = sum(precisions) / sum(x > 0 for x in total_instances)
            client_mAP.append(mAP)
            logfile.write('client:{} patient{}\n'.format(client_name[i], patient_name))
            logfile.write('  TP:{} FP:{}\n'.format(patient_tp, patient_fp))
            logfile.write('  FP/slice:{} Sensitivity:{}  Precision:{}\n'.format(patient_fp / num_slices,
                                                                                patient_tp / num_annotations,
                                                                                patient_tp / (patient_tp + patient_fp)))
            logfile.write('  mAP:{}\n'.format(mAP))

        client_total_recall = client_tps / client_num_annotations
        client_total_precision = client_tps / (client_tps + client_fps)
        clint_total_fp_slice = client_fps / client_num_slices

        client_avg_recall = sum(client_recall) / len(test_data_list[i])
        client_avg_precision = sum(client_precision) / len(test_data_list[i])
        client_avg_fp_slice = sum(client_fp_slice) / len(test_data_list[i])
        avg_mAP = sum(client_mAP) / len(test_data_list[i])

        # calc confidence interval
        client_mean_precision = np.mean(np.asanyarray(client_precision))
        client_std_precision = np.std(np.asanyarray(client_precision))
        client_se_precision = client_std_precision / np.sqrt(len(test_data_list))
        client_p_value_precision = 1.96 * client_se_precision

        client_mean_recall = np.mean(np.asanyarray(client_recall))
        client_p_value_recall = 1.96 * (np.std(np.asanyarray(client_recall)) / np.sqrt(len((test_data_list))))

        client_mean_map = np.mean(np.asanyarray(client_mAP))
        client_p_value_map = 1.96 * (np.std(np.asanyarray(client_mAP)) / np.sqrt(len((test_data_list))))


        print('------{}------'.format(client_name[i]))
        print('   total:')
        print('   # TP:{}   FP{}'.format(client_tps, client_fps))
        print('   # FP/slice:{:.4f} Sensitivity:{:.5f}  Precision:{:.5f}'.format(clint_total_fp_slice,
                                                                                 client_total_recall,
                                                                                 client_total_precision))
        print('   average:')
        print('   # TP:{}   FP{}'.format(client_tps, client_fps))
        print('   # FP/slice:{:.4f} Sensitivity:{:.5f}  Precision:{:.5f}'.format(client_avg_fp_slice, client_avg_recall,
                                                                                 client_avg_precision))
        print('   # mAP:{:.5f}'.format(avg_mAP))
        print('   # Sensitivity:[{:.4f}+-{:.4f}]   Precision:[{:.4f}+-{:.4f}] mAP:[{:.4f}+-{:.4f}]'.format(
            client_mean_recall, client_p_value_recall, client_mean_precision, client_p_value_precision, client_mean_map,
            client_p_value_map))

        logfile.write('  total over client:{}\n'.format(client_name[i]))
        logfile.write('  TP:{} FP:{}\n'.format(client_tps, client_fps))
        logfile.write('  FP/slice:{} Sensitivity:{}  Precision:{}\n'.format(clint_total_fp_slice, client_total_recall,
                                                                            client_total_precision))
        logfile.write('  Average over client:{}\n'.format(client_name[i]))
        logfile.write('  TP:{} FP:{}\n'.format(client_tps, client_fps))
        logfile.write('  FP/slice:{} Sensitivity:{}  Precision:{}\n'.format(client_avg_fp_slice, client_avg_recall,
                                                                            client_avg_precision))
        logfile.write('  mAP:{}\n'.format(avg_mAP))
        logfile.write('*********************************')

    logfile.flush()
    logfile.close()

if __name__ == '__main__':
    main()
