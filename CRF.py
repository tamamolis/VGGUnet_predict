import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels
from skimage.color import gray2rgb
import os
import cv2 as cv

Building = [0, 0, 255]
Grass = [0, 255, 0]
Development = [255, 255, 0]  # стройка
Concrete = [255, 255, 255]  # бетон
Roads = [0, 255, 255]
NotAirplanes = [252, 0, 252]
Unlabelled = [255, 0, 0]

height = 416
width = 608
n_classes = 7

legend_list = [Building, Grass, Development, Concrete, Roads, NotAirplanes, Unlabelled]
# legend_list = [Development, Grass, Concrete, Unlabelled, Building]


def visualize(temp):
    color = np.array([Building, Grass, Development, Concrete, Roads, NotAirplanes, Unlabelled])
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = color[l, 0]
        g[temp == l] = color[l, 1]
        b[temp == l] = color[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)
    rgb[:, :, 1] = (g / 255.0)
    rgb[:, :, 2] = (b / 255.0)
    return rgb


def euclidean_metric(inp):
    # евклидова метрика
    total = 100000
    index = -1

    for legend in legend_list:
        new_total = np.linalg.norm(np.array(legend) - inp)
        # print(new_total)
        if new_total < total:
            total = new_total
            index = legend_list.index(legend)
    return index


def table(img):

    h = np.shape(img)[0]
    w = np.shape(img)[1]
    new_array = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # print(img[i][j])
            buf = euclidean_metric(img[i][j])
            # print(buf)
            new_array[i][j] = buf
    return new_array


def crf(original_image, annotated_image, output_image, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale

    if len(annotated_image.shape) < 3:
        annotated_image = gray2rgb(annotated_image)

    annotated_label = table(annotated_image)
    print('table done')

    print(np.shape(annotated_image))
    colors, labels = np.unique(annotated_label, return_inverse=True)

    colors = np.array([0, 1, 2, 3, 4]) # было до 6!
    print(colors)
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat))

    f = open('labels.txt', 'w')
    for lines in labels:
        f.write(str(lines))
    f.close()

    labels = np.array(labels)
    if use_2d:
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        print('labels: ', labels, len(labels), np.shape(labels))
        print('n_lables: ', n_labels)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(5, 5), compat=4, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(40, 40), srgb=(8, 8, 8), rgbim=original_image,
                               compat=3,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    MAP = np.argmax(Q, axis=0)

    rgb = visualize(MAP.reshape(np.shape(original_image)[0], np.shape(original_image)[1]))
    imsave(output_image, rgb)


def stretch(path, path_save):
    files = os.listdir(path)
    for file in files:
        img = cv.imread(path + file)
        print(path + file)
        res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        # height, width = img.shape[:2]
        # res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
        name = 'resize.png'
        path_save = path_save + name
        cv.imwrite(path_save, res)


def crop_orig(path_orig, path_seg):
    img = cv.imread(path_seg + 'resize.png')
    print(path_seg + 'resize.png')
    h, w, c = img.shape
    print(h, w, c)

    files_orig = os.listdir(path_orig)
    img_orig = cv.imread(path_orig + files_orig[0])
    crop_img = img_orig[0: h, 0: w]
    cv.imwrite(path_seg + 'crop.png', crop_img)


if __name__ == '__main__':
    orig = 'data/crop/crop.png'
    seg = 'data/crop/resize.png'

    image = imread(orig)
    seg_image = imread(seg)

    crf(image, seg_image, "data/crop/result_crf_3.jpg")