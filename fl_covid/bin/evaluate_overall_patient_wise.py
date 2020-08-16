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
import warnings

warnings.simplefilter("ignore")
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
from ..utils.visualization import draw_box, label_color, draw_caption
from fl_covid.bin.train_fed import create_models

# Warm up for the model inference time counting
WARM_UP = 2


def draw_label_hit(image, box, caption):
    """
    Draw a caption above the box in an image.
    :param image: The image to draw on.
    :param box: A list of 4 elements (x1, y1, x2, y2).
    :param caption: String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0]+5, b[3] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0]+5, b[3] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """
    Draw annotations on image
    :param image: The image to draw on.
    :param annotations: A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
    :param color: The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
    :param label_to_name: (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c, thickness=1)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, slice_id=None, bbox_writer=None, score_threshold=0.4):
    """
    Draw detections on image.
    :param image: The image to draw on.
    :param boxes: A [N, 4] matrix (x1, y1, x2, y2).
    :param scores: A list of N classification scores.
    :param labels: A list of N labels.
    :param color: The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
    :param label_to_name: (optional) Functor for mapping a label to a name.
    :param slice_id: The i-th slice
    :param bbox_writer: csv writer to write out the box position to csv file
    :param score_threshold: Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]
    for i in selection:
        c = color if color is not None else label_color(labels[i])

        if bbox_writer is not None and slice_id is not None:
            tar_path = 'slice_{}.png'.format(slice_id)
            b = np.array(boxes[i, :]).astype(int)
            bbox_writer.writerow([tar_path]+ [b[0],b[1],b[2],b[3]]+['lesion'])

        draw_box(image, boxes[i, :], color=[245,245,245],thickness=2)

        # draw labels
        # caption = (label_to_name(labels[i]) if label_to_name else str(labels[i])) + ': {0:.2f}'.format(scores[i])
        # draw_caption(image, boxes[i, :], caption)


def read_h5(img_path):
    """
    read the file in h5 format
    :param img_path:  path of h5 files
    :return:  np.array
    """
    with h5py.File(img_path, "r") as hf:
        arr = hf['arr'][:]
    return arr


def fp_reduce(fp_detections):
    """
    reduce the fp bbox contained by bigger one
    :param fp_detections:
    :return: array in shape (slices,boxes)
    """
    deleted_fp_detections = [[] for i in range(len(fp_detections))]
    cnt = 0
    for i in range(len(fp_detections)):
        slice_detection = fp_detections[i]
        slice_deleted_fp = np.empty([0,5],dtype=np.float32)
        if len(slice_detection) <= 1:
            continue
        else:
            iou, iou_self, iou_other = compute_overlap(slice_detection,slice_detection)

            x,y =  np.where(iou_self>0.7)
            for j in range(len(x)):
                if x[j] == y[j]:
                    continue
                else:
                    slice_deleted_fp = np.concatenate((slice_deleted_fp, np.expand_dims(slice_detection[x[j],:],axis=0)))
        deleted_fp_detections[i] = slice_deleted_fp
    return deleted_fp_detections


def draw_colorful_result(
    args,
    client_name,
    patient_name,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """Draw the result with colored bounding box
    """

    def _parse(value, function, fmt):
        """Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))
    if args.reduce_fp:
        sign = 'fp_reduced_'
    else:
        sign=''
    bbox_result_path = os.path.join(save_path,'{}_{}_score_thres_{}_bbox.csv'.format(client_name, patient_name, score_threshold))
    anno_result_path = os.path.join(save_path,'{}_{}_score_thres_{}_anno.csv'.format(client_name, patient_name, score_threshold))
    all_annotations_img_path = np.load(os.path.join(save_path, '{}_{}_annotations_img_path.npy'.format(client_name, patient_name)), allow_pickle=True)

    # prepare annotation result
    anno_result = {}
    annos = open(anno_result_path, 'r')
    classes = {'lesion': 0}
    for line, row in enumerate(annos):
        splits = row.split(',')
        try:
            img_file, x1, y1, x2, y2, class_name, hit_cnt = splits
            hit_cnt = hit_cnt.replace('\n', '')

        except ValueError:
            raise ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))
        if img_file not in anno_result:
            anno_result[img_file] = []

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
        hit_cnt = _parse(hit_cnt, int, 'line {}: malformed hit count: {{}}'.format(line))

        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if str(class_name) not in classes:
            raise ValueError(
                'line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        anno_result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name, 'hit_cnt':hit_cnt})

    # prepare prediction bbox result
    bbox_result = {}
    bboxs = open(bbox_result_path, 'r')
    classes = {'lesion': 0}
    for line, row in enumerate(bboxs):
        splits = row.split(',')
        try:
            img_file, x1, y1, x2, y2, class_name, score, box_type = splits
            box_type = box_type.replace('\n', '')

        except ValueError:
            raise ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))
        if img_file not in bbox_result:
            bbox_result[img_file] = []

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if str(class_name) not in classes:
            raise ValueError(
                'line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        bbox_result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name, 'score':score, 'box_type': str(box_type)})

    detection_out = np.zeros([len(all_annotations_img_path), 512, 512, 3])

    for i in tqdm(range(len(all_annotations_img_path)), desc='Drawing colorful {} result  on {} {}: '.format(sign, client_name, patient_name)):
        img_path = all_annotations_img_path[i]
        raw_img = read_h5(img_path)
        image = raw_img.copy()

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        if img_path in anno_result:
            for anno_index in range(len(anno_result[img_path])):
                # draw annotation
                hit_cnt = anno_result[img_path][anno_index]['hit_cnt']
                caption = '{}'.format(hit_cnt)
                anno_box = [anno_result[img_path][anno_index]['x1'], anno_result[img_path][anno_index]['y1'], anno_result[img_path][anno_index]['x2'],anno_result[img_path][anno_index]['y2']]
                draw_label_hit(image, anno_box , caption)
                draw_box(image, anno_box, color=[0,255,0], thickness=1)

        if img_path in bbox_result:
            for bbox_index in range(len(bbox_result[img_path])):
                pred_box = [bbox_result[img_path][bbox_index]['x1'], bbox_result[img_path][bbox_index]['y1'], bbox_result[img_path][bbox_index]['x2'],bbox_result[img_path][bbox_index]['y2']]
                box_type = str(bbox_result[img_path][bbox_index]['box_type'])
                score = float(bbox_result[img_path][bbox_index]['score'])

                if box_type == 'max_overlap':
                    box_color = [31, 0, 255]
                elif box_type == 'assigned_pre':
                    box_color =[184, 0, 255]
                elif box_type == 'assigned_gt':
                    box_color = [139, 69, 19]
                elif box_type == 'fp':
                    box_color = [225, 0, 0]
                else:
                    raise ValueError("Unknown box type :{}".format(box_type))

                draw_box(image, pred_box, color=box_color, thickness=1)
                caption = ('{0:.2f}'.format(score))
                draw_caption(image, pred_box, caption)

        detection_out[i, :, :] = image
    print('Writing colorful results on {} {}...'.format(client_name, patient_name))
    detection_out = sitk.GetImageFromArray(detection_out)
    sitk.WriteImage(detection_out, os.path.join(save_path, '{}_{}_colorful_detection_{}result.nii.gz'.format(client_name, patient_name, sign)))


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
    """
    filter bboxes out of the lung using the lung segmentation image
    :param bboxes: array of boxes ([N][4])
    :param scores_sort: sorted prediction score
    :param seg: the segmentation image
    :return:
    """
    image_boxes = bboxes
    inner = np.asarray([],dtype=np.bool)

    for i in range(image_boxes.shape[0]):
        x1 = int(image_boxes[i][0])
        y1 = int(image_boxes[i][1])
        x2 = int(image_boxes[i][2])
        y2 = int(image_boxes[i][3])
        x1 = 511 if x1 > 511 else x1
        y1 = 511 if y1 > 511 else y1
        x2 = 511 if x2 > 511 else x2
        y2 = 511 if y2 > 511 else y2

        if (seg[y1,x1,:] == 0).all() and (seg[y2,x2,:] == 0).all() and  (seg[y1,x2,:] == 0).all() and  (seg[y2,x1,:] == 0).all():
            inner = np.append(inner,False)
        else:
            inner = np.append(inner, True)

    scores_sort = scores_sort[inner]

    return scores_sort


def _print_detections_to_npy(args, generator, model, client_idx, client_name, patient_name, score_threshold=0.05, max_detections=100, save_path=None):
    """
    get the detection results in array format and saved
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    detection_out = np.zeros([generator.size(),512,512])
    mask_out = np.zeros([generator.size(),512,512])

    inference_times = []
    index = 0

    for i in tqdm(range(generator.size()), desc='Running network on {}...: '.format(client_name)):
        index += 1
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # after warming up , start logging the model inference time
        if index > WARM_UP:
            start = time.clock()

        # run network
        boxes, scores, labels, masks, _ = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        if index > WARM_UP:
            inference_times.append(time.clock() - start)

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > -1)[0]
        if type(scores) is not np.ndarray:
            scores = scores.numpy()
            boxes = boxes.numpy()
            labels = labels.numpy()
            masks = masks.numpy()
        # select those scores
        scores = scores[0][indices]
        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]
        image_boxes = boxes[0, indices[scores_sort], :]

        # filter boxes out of lung
        if args.lung_filter:
            if args.site.lower() == 'internal':
                lung_filter_path = './data/internal/internal_test/lung_seg_h5/'
            elif args.site.lower() == 'external1' or args.site.lower() == 'external2' or args.site.lower() == 'external3':
                lung_filter_path = './data/{}/lung_seg_h5/'.format(args.site.lower())
            else:
                raise ValueError('Unknown site {}'.format(args.site))
            # TODO: change to the relative path before releasing

            img_path = generator.image_path(i)

            patient = img_path.split('/')[-2]
            slice_idx = img_path.split('/')[-1].replace('slice_', '').replace('.h5', '')
            # seg_path = os.path.join(lung_filter_path,'{}_slice_{}.png').format(patient,slice_idx)
            seg_path = os.path.join(lung_filter_path,'slice_{}.h5').format(slice_idx)
            with h5py.File(seg_path, "r") as hf:
                seg = hf['arr'][:]
                seg = np.expand_dims(seg,axis=-1)
            # seg = cv2.imread(seg_path)

            filter_mask = np.zeros([1,512,512,1])
            filter_mask[0, np.where(seg == 255)[0], np.where(seg == 255)[1], 0] = masks[0, np.where(seg == 255)[0], np.where(seg == 255)[1], 0]
            scores_sort = _seg_filter(image_boxes, scores_sort, seg)
            image_boxes = boxes[0, indices[scores_sort], :]

        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        if args.save_result == 1:
            img_path = generator.image_path(i)
            img_path = img_path.replace('h5_normalize', 'h5')

            with h5py.File(img_path, "r") as hf:
                h5_raw_image = hf['arr'][:]
            draw_detections(h5_raw_image, image_boxes, image_scores, image_labels, slice_id=i, score_threshold=args.score_threshold)
            detection_out[i, :, :] = h5_raw_image[:,:,1]

            masks[masks < args.segmentation_threshold] = 0
            # set segmentation scoremap area out of the lung to 0
            filter_mask[filter_mask < args.segmentation_threshold] = 0
            filter_mask = cv2.resize(np.squeeze(np.uint8(filter_mask * 255)), (512, 512))

            mask_out[i, :, :] = filter_mask

    print('Inference time (ms): {:.2f}'.format(np.average(inference_times) * 1000), flush=True)

    if save_path is not None and args.save_result == 1:
        print('Inference done, Saving visual Results, Please wait...')
        visual_res_path = os.path.join(save_path,'visual_results')
        if not os.path.exists(visual_res_path):
            os.makedirs(visual_res_path)

        detection_out = sitk.GetImageFromArray(detection_out)
        sitk.WriteImage(detection_out, os.path.join(visual_res_path, '{}_{}_detection_result.nii.gz'.format(client_name, patient_name)))

        mask_out = sitk.GetImageFromArray(mask_out)
        sitk.WriteImage(mask_out, os.path.join(visual_res_path, '{}_{}_masks_result.nii.gz'.format(client_name, patient_name)))

    statistical_res_path = os.path.join(save_path, 'statistical_results')
    if not os.path.exists(statistical_res_path):
        os.makedirs(statistical_res_path)
    np.save(os.path.join(statistical_res_path, '{}_{}_prediction.npy'.format(client_name, patient_name)), all_detections)
    all_annotations, all_annotations_img_path = _get_annotations_and_img_path(generator)
    np.save(os.path.join(statistical_res_path, '{}_{}_annotations.npy'.format(client_name, patient_name)), all_annotations)
    # np.save(os.path.join(save_path, '{}_{}_annotations_img_path.npy'.format(client_name, patient_name)), all_annotations_img_path)

    return 0


def evaluate_from_npy(
    args,
    client_name,
    patient_name,
    # resolution,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
):
    """
    Evaluate a given dataset using a given model.
    :param iou_threshold: The threshold used to consider when a detection is positive or negative.
    :param score_threshold: The score confidence threshold to use for detections.
    :param max_detections: The maximum number of detections to use per image.
    :param save_path: The path to save detection results
    :param verbose: Whether to print out the information
    """
    verbose = args.verbose
    # gather all detections and annotations
    if args.reduce_fp:
        all_detections = np.load(os.path.join(save_path, 'statistical_results/{}_{}_prediction_fp_reduced.npy'.format(client_name, patient_name)), allow_pickle=True)
    else:
        all_detections = np.load(os.path.join(save_path, 'statistical_results/{}_{}_prediction.npy'.format(client_name, patient_name)), allow_pickle=True)
    all_annotations          = np.load(os.path.join(save_path, 'statistical_results/{}_{}_annotations.npy'.format(client_name, patient_name)), allow_pickle=True)
    all_fp_detections = [[] for j in range(all_annotations.shape[0])]

    average_precisions = {}

    # only have 1 label (lesion or not)
    for label in range(1):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        false_negatives = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        fp_all = {}
        tp_all = {}
        hitter_all = {}

        for i in range(all_annotations.shape[0]):
            detections           = all_detections[i][label]
            detections           = detections[detections[:, -1] >= score_threshold]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]

            slice_fp_detections = np.empty([0, 5], dtype=np.float32)
            hitter = np.zeros(annotations.shape[0])
            fp = 0
            for d in detections:
                # no detections
                if annotations.shape[0] == 0:
                    continue
                # calculate the iou in 3 different methods
                ious, overlaps_pre_arr, overlaps_gt_arr = compute_overlap(np.expand_dims(d, axis=0), annotations)

                assigned_annotation = np.argmax(ious, axis=1)
                max_overlap         = ious[0, assigned_annotation]

                if max_overlap >= iou_threshold:
                    if hitter[assigned_annotation] == 0:
                        scores = np.append(scores, d[4])
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)

                    hitter[assigned_annotation] += 1
                else:
                    assigned_annotation_pre = np.where(overlaps_pre_arr > iou_threshold)
                    assigned_annotation_gt = np.where(overlaps_gt_arr > iou_threshold)

                    if len(assigned_annotation_pre[0]) > 0:
                        for index in assigned_annotation_pre[1]:
                            # first time hit
                            if hitter[index] == 0:
                                scores = np.append(scores, d[4])
                                false_positives = np.append(false_positives, 0)
                                true_positives = np.append(true_positives, 1)

                            hitter[index] += 1

                    if len(assigned_annotation_gt[0]) > 0:

                        for index in assigned_annotation_gt[1]:
                            if hitter[index] == 0:
                                scores = np.append(scores, d[4])
                                false_positives = np.append(false_positives, 0)
                                true_positives = np.append(true_positives, 1)

                            hitter[index] += 1

                    if len(assigned_annotation_pre[0]) + len(assigned_annotation_gt[0]) == 0:
                        fp += 1
                        scores = np.append(scores, d[4])
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

                        slice_fp_detections = np.concatenate((slice_fp_detections, np.expand_dims(d, axis=0)), axis=0)

            all_fp_detections[i] = slice_fp_detections

            # Get the tp/fp by hitter count
            hitter_all[i] = hitter
            tp_all[i] = len(np.where(hitter > 0)[0])
            fp_all[i] = fp

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        before_reduce_fp = 0
        for i in range(len(all_fp_detections)):
            before_reduce_fp += len(all_fp_detections[i])

        # reduce fp in detections
        deleted_all_fp_detections = fp_reduce(all_fp_detections)
        deleted_fp_num = 0

        # after_reduce_fp = 0
        for i in range(len(deleted_all_fp_detections)):
            deleted_fp_num += len(deleted_all_fp_detections[i])

        TP_ALL = 0
        FP_ALL = 0

        for key in tp_all.keys():
            TP_ALL += tp_all[key]
        for key in fp_all.keys():
            FP_ALL += fp_all[key]

        FP_ALL -= deleted_fp_num

        FP_slice = FP_ALL / all_annotations.shape[0]
        Sensitivity = TP_ALL / num_annotations
        Precision = TP_ALL / (TP_ALL + FP_ALL) if (TP_ALL + FP_ALL) > 0 else 0

        result_list = [TP_ALL,FP_ALL, Sensitivity, Precision, FP_slice]


        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        if verbose==1:
            print('------{} {}------'.format(client_name, patient_name))
            print('   # TP:{}   FP{} '.format(TP_ALL, FP_ALL))
            print('   # FP/Slice:{:.4f} Sensitivity:{:.5f}  Precision:{:.5f}'.format(FP_slice, Sensitivity, Precision))

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

        all_detections_reduce_fp = all_detections.copy()
        for i in range(len(all_detections_reduce_fp)):
            slice_detections = all_detections_reduce_fp[i][0]

            deleted_slice_fp = deleted_all_fp_detections[i]

            new_slice_detections = np.empty([0,5],dtype=np.float32)

            if len(deleted_slice_fp) == 1:
                fp_indices =  np.where(slice_detections==deleted_slice_fp)
                x,y = fp_indices
                deleted_idx = set(x)
                all_idx = set(list(range(len(slice_detections))))
                remain_idx = all_idx.difference(deleted_idx)
                remain_idx = list(remain_idx)
                for idx in remain_idx:
                    new_slice_detections = np.concatenate((new_slice_detections, np.expand_dims(slice_detections[idx],axis=0)),axis=0)
                assert len(remain_idx) == len(new_slice_detections)

            elif len(deleted_slice_fp) > 1:
                all_deleted_idx = []
                for each_deleted_slice_fp in deleted_slice_fp:
                    x, y = np.where(slice_detections == each_deleted_slice_fp)
                    deleted_idx = set(x)
                    all_deleted_idx.append(list(deleted_idx)[0])

                all_idx = set(list(range(len(slice_detections))))
                all_deleted_idx = set(all_deleted_idx)
                remain_idx = all_idx.difference(all_deleted_idx)
                remain_idx = list(remain_idx)

                for idx in remain_idx:
                    new_slice_detections = np.concatenate((new_slice_detections, np.expand_dims(slice_detections[idx], axis=0)), axis=0)
                assert len(remain_idx) == len(new_slice_detections)
            else:
                continue

            all_detections_reduce_fp[i][0] = new_slice_detections
    return average_precisions, result_list, num_annotations, all_annotations.shape[0]


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    parser.add_argument('--model', help='Path to RetinaNet model.', default=None)
    parser.add_argument('--weights', help='only load weights.', default=None)
    parser.add_argument('--site', help='which site used to test, options [Internal | External1 | External3]', default='Internal')
    parser.add_argument('--nii',              help='path to nii files.')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='vgg19_nl')
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
    parser.add_argument('--lung-filter',        help='Path for lung seg filter images', default=True, action='store_true')
    parser.add_argument('--draw-colorful', help='draw difficult type of predict with color', default=False, action='store_true')
    parser.add_argument('--reduce-fp', help='reduce fp, must use after completing first evaluation', default=False, action='store_true')
    parser.add_argument('--score-loop', help='reduce fp, must use after completing first evaluation', default=False, action='store_true')
    parser.add_argument('--verbose', help='whether to print out info', default=False, action='store_true')
    parser.add_argument('--log',        help='Path for saving log file', default=None)
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=512)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=512)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).', default='./fl_covid/anchors4.ini')
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

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    anno_base_dir = './data'
    args.classes = './fl_covid/utils/mapping.csv'

    # print model summary
    # print(model.summary())

    # create the generator
    # Internal
    if args.site.lower() == 'internal':
        client_name = ['Internal']
        data_path = ['internal/internal_test/h5_normalize']
        internal = ['internal_test.csv']
        test_data_list = [internal]

    # External-1
    elif args.site.lower() == 'external1':
        client_name = ['External1']
        data_path = ['external1/h5_normalize']
        external_1 = ['external1_test.csv']
        test_data_list = [external_1]

    # External-2
    elif args.site.lower() == 'external2':
        client_name = ['External2']  # public
        data_path = ['external2/h5_normalize']
        external_2 = ['external2_test.csv']
        test_data_list = [external_2]

    # External-3
    elif args.site.lower() == 'external3':
        client_name = ['External3']  # SZ
        data_path = ['external3/h5_normalize']
        external_3 = ['external3_test.csv']
        test_data_list = [external_3]
    else:
        raise ValueError('Unknown site {}'.format(args.site))

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
    log_path = args.log if args.log else './evaluate_overall_patient_wise_log.txt'
    logfile = open(log_path,'a')

    # save prediction to npy
    if args.get_predicted_bbox == 1:
        logfile.write('*********************************\n')
        logfile.write('*{}*\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('Save prediction of model:{} to .npy file\n'.format(args.model))

        backbone = models.backbone(args.backbone)

        # optionally load anchor parameters
        anchor_params = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)

        # load model
        if args.model is not None:
            print('Loading model, this may take a second...', flush=True)
            model = models.load_model(args.model, backbone_name=args.backbone)
        elif args.weights is not None:
            weights = args.weights

            print('Creating model and Loading weights, this may take a second...', flush=True)
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
        else:
            raise ValueError("You have to specify a model")

        # Convert the model to predict the position of bbox directly
        model = models.convert_model(model, anchor_params=anchor_params)

        # create generator
        generators = {}
        for i in range(len(client_name)):
            print('---Client {}---'.format(client_name[i]), flush=True)
            for j in range(len(test_data_list[i])):
                args.annotations = os.path.join(anno_base_dir, data_path[i], test_data_list[i][j])
                print('Validation csv {}'.format(args.annotations), flush=True)
                generator = create_generator(args)
                if client_name[i] not in generators:
                    generators[client_name[i]] = []
                    generators[client_name[i]].append(generator)
                else:
                    generators[client_name[i]].append(generator)

        for i in range(len(generators)):
            for j in range(len(generators[client_name[i]])):
                logfile.write('Writing client {} patient {} prediction results to .npy... \n'.format(client_name[i], patient_names[client_name[i]]))
                generator = generators[client_name[i]][j]
                patient_name = patient_names[client_name[i]][j]
                _print_detections_to_npy(
                    args,
                    generator,
                    model,
                    client_idx=i,
                    client_name=client_name[i],
                    patient_name = patient_name,
                    score_threshold=args.score_threshold,
                    max_detections=args.max_detections,
                    save_path=args.save_path,
                )
        logfile.write('Finish writing \n'.format(i))
        logfile.write('*********************************')
        sys.exit(0)

    # score loop and write precision-recall pairs
    if args.score_loop:
        all_client_recall, all_client_precision, all_client_fpr, all_client_map = [], [] ,[], []
        all_patient_recall, all_patient_precision, all_patient_fpr, all_patient_map = [], [],[], []
        for i in range(len(client_name)):
            roc = open(os.path.join(args.save_path, 'roc_data_iou_{}_{}.csv'.format(args.iou_threshold, client_name[i])), 'w', newline='')
            roc_writer = csv.writer(roc, delimiter=',')
            roc_writer.writerow(['client_name', 'FP rate','Score threshold', 'Sensitivity','CI','Precision','CI','mAP','CI'])

            print('client {}'.format(client_name[i]), flush=True)
            # fp_rate = 1000.
            recall_flag = 1000.
            precision_flag = 0.

            for score in range(0, 99):
                client_tps, client_fps, client_num_annotations, client_num_slices = 0., 0., 0., 0.
                client_precision, client_recall, client_fp_slice = [], [], []
                client_mAP = []

                score_threshold = score / 100
                print('Using score:{}'.format(score_threshold), flush=True)

                for j in range(len(test_data_list[i])):
                    patient_name = patient_names[client_name[i]][j]

                    if not os.path.exists(os.path.join(args.save_path, 'roc_data_iou_{}_{}_{}.csv'.format(args.iou_threshold, client_name[i], patient_name))):
                        roc_per_patient = open(os.path.join(args.save_path, 'roc_data_iou_{}_{}_{}.csv'.format(args.iou_threshold, client_name[i], patient_name)), 'w', newline='')
                        roc_writer_per_patient = csv.writer(roc_per_patient, delimiter=',')
                        roc_writer_per_patient.writerow(['patient_name', 'FP rate', 'Score threshold', 'Sensitivity', 'Precision','mAP'])
                    else:
                        roc_per_patient = open(
                            os.path.join(args.save_path, 'roc_data_iou_{}_{}_{}.csv'.format(args.iou_threshold, client_name[i], patient_name)),
                            'a', newline='')
                        roc_writer_per_patient = csv.writer(roc_per_patient, delimiter=',')

                    average_precisions, new, num_annotations, num_slices = evaluate_from_npy(
                        args,
                        client_name=client_name[i],
                        patient_name=patient_name,
                        iou_threshold=args.iou_threshold,
                        score_threshold=score_threshold,
                        max_detections=args.max_detections,
                        save_path=args.save_path,
                    )
                    # calc average
                    client_recall.append(new[2])
                    client_precision.append(new[3])
                    client_fp_slice.append(new[4])

                    # add together and calc (total)
                    patient_tp = new[0]
                    patient_fp = new[1]
                    client_tps += patient_tp
                    client_fps += patient_fp
                    client_num_annotations += num_annotations
                    client_num_slices += num_slices

                    # collect all patient metrics
                    all_patient_recall.append(new[2])
                    all_patient_precision.append(new[3])
                    all_patient_fpr.append(new[4])

                    # print evaluation
                    total_instances = []
                    precisions = []
                    for label, (average_precision, num_annotations) in average_precisions.items():
                        total_instances.append(num_annotations)
                        precisions.append(average_precision)

                    if sum(total_instances) == 0:
                        print('No test instances found.')
                        return

                    mAP = sum(precisions) / sum(x > 0 for x in total_instances)
                    client_mAP.append(mAP)

                    all_patient_map.append(mAP)

                    roc_writer_per_patient.writerow(
                        ['internal_client_wise'] + [round(new[4], 4)] + [score_threshold] + [round(new[2], 5)] + [
                            round(new[3], 5)] + [round(mAP, 5)])
                    roc_per_patient.flush()
                    roc_per_patient.close()

                client_avg_recall = sum(client_recall) / len(test_data_list[i])
                client_avg_precision = sum(client_precision) / len(test_data_list[i])
                client_avg_fp_slice = sum(client_fp_slice) / len(test_data_list[i])
                client_avg_mAP = sum(client_mAP) / len(test_data_list[i])


                all_client_recall.append(client_avg_recall)
                all_client_precision.append(client_avg_precision)
                all_client_fpr.append(client_avg_fp_slice)
                all_client_map.append(client_avg_mAP)

                # calc confidence interval
                client_mean_precision = np.mean(np.asanyarray(client_precision))
                client_std_precision = np.std(np.asanyarray(client_precision))
                client_se_precision = client_std_precision / np.sqrt(len(test_data_list[i]))
                client_p_value_precision = 1.96*client_se_precision

                client_mean_recall= np.mean(np.asanyarray(client_recall))
                client_p_value_recall= 1.96 *(np.std(np.asanyarray(client_recall)) / np.sqrt(len((test_data_list[i]))))

                client_mean_map = np.mean(np.asanyarray(client_mAP))
                client_p_value_map = 1.96 * (np.std(np.asanyarray(client_mAP)) / np.sqrt(len((test_data_list[i]))))

                if ((recall_flag - client_avg_recall) < 0.0001 and (precision_flag - client_avg_precision) < 0.0001) or (
                        recall_flag < client_avg_recall or precision_flag > client_avg_precision):
                    print('CONTIENUE')
                    continue
                else:
                    recall_flag = client_avg_recall
                    precision_flag = client_avg_precision

                    roc_writer.writerow(
                        [client_name[i]] + [round(client_avg_fp_slice, 4)] + [score_threshold] +
                        [round(client_avg_recall, 5)] + ['[{}+-{}]'.format(client_mean_recall, client_p_value_recall)] +
                        [round(client_avg_precision, 5)] + ['[{}+-{}]'.format(client_mean_precision, client_p_value_precision)] +
                        [round(client_avg_mAP,5)] + ['[{}+-{}]'.format(client_mean_map, client_p_value_map)])

                print('------{}------'.format(client_name[i]))
                print('   average:')
                print('   # TP:{}   FP{}'.format(client_tps, client_fps))
                print('   # FP/slice:{:.4f} Sensitivity:{:.5f}  Precision:{:.5f}'.format(client_avg_fp_slice,
                                                                                         client_avg_recall,
                                                                                         client_avg_precision))
                print('   # mAP:{:.5f} {} {}'.format(client_avg_mAP, sum(client_mAP), len(test_data_list[i])))
                print('   # Sensitivity:[{:.4f}+-{:.4f}]   Precision:[{:.4f}+-{:.4f}] mAP:[{:.4f}+-{:.4f}]'.format(
                    client_mean_recall, client_p_value_recall, client_mean_precision, client_p_value_precision,
                    client_mean_map, client_p_value_map))

    else:
        # evaluate from npy
        logfile.write('*********************************\n')
        logfile.write('*{}*\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('Evaluate model:{} from .npy\n'.format(args.model))
        logfile.write('thresshold:{}\n'.format(args.score_threshold))

        for i in range(len(client_name)):
            print('client {}'.format(client_name[i]))
            client_tps, client_fps, client_num_annotations, client_num_slices = 0., 0., 0., 0.
            client_precision, client_recall, client_fp_slice = [], [], []
            client_mAP = []
            for j in range(len(test_data_list[i])):
                patient_name = patient_names[client_name[i]][j]
                average_precisions, new, num_annotations, num_slices = evaluate_from_npy(
                    args,
                    client_name=client_name[i],
                    patient_name=patient_name,
                    # resolution=test_data_resolution[i][j],
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

                # print evaluation
                total_instances = []
                precisions = []
                for label, (average_precision, num_annotations) in average_precisions.items():
                    total_instances.append(num_annotations)
                    precisions.append(average_precision)

                if sum(total_instances) == 0:
                    print('No test instances found.')
                    return

                if args.weighted_average:
                    print('    mAP: {:.5f}'.format(
                        sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
                else:
                    print('    mAP: {:.5f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

                mAP = sum(precisions) / sum(x > 0 for x in total_instances)
                client_mAP.append(mAP)
                logfile.write('client:{} patient{}\n'.format(client_name[i], patient_name))
                logfile.write('  TP:{} FP:{}\n'.format(patient_tp, patient_fp))
                logfile.write('  FP/slice:{} Sensitivity:{}  Precision:{}\n'.format(patient_fp/num_slices, patient_tp/num_annotations, patient_tp/(patient_tp + patient_fp) if patient_tp + patient_fp >0 else 0))
                logfile.write('  mAP:{}\n'.format(mAP))

            logfile.write('*********************************')

    logfile.flush()
    logfile.close()

if __name__ == '__main__':
    main()
