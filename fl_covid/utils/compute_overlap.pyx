# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np


def compute_overlap(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] ious = np.zeros((N, K), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] overlaps_gt = np.zeros((N, K), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] overlaps_pre = np.zeros((N, K), dtype=np.float64)
    cdef double iw, ih, gt_area, pre_area
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        gt_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            pre_area = (
                (boxes[n, 2] - boxes[n, 0] + 1) *
                (boxes[n, 3] - boxes[n, 1] + 1)
            )
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        gt_area - iw * ih
                    )
                    ious[n, k] = iw * ih / ua
                    overlaps_gt[n,k] = iw * ih / gt_area
                    overlaps_pre[n,k] = iw * ih / pre_area
    return ious, overlaps_pre, overlaps_gt
