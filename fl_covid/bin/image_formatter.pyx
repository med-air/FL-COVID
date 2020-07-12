import cython

@cython.boundscheck(False)
cpdef long long[:, :] format_image(int scalar, float multiplier, long long [:, :] image):
     cdef int x, y, w, h

     h = image.shape[0]
     w = image.shape[1]

     for y in range(0, h):
        for x in range(0, w):
            image[x, y] = int(min(max((image[x, y] + scalar) / multiplier, 0), 255))

     return image