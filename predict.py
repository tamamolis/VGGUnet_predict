import glob
import numpy as np
import cv2
from model import set_keras_backend, VGGUnet
import keras.models as models

Building = [0, 0, 255]
Grass = [0, 255, 0]
Development = [0, 255, 255]  # стройка, было [255, 255, 0], тогда оно красилось ГОЛУБЫМ!!!
Concrete = [255, 255, 255]  # бетон
Roads = [0, 255, 255]
NotAirplanes = [252, 40, 252]
Unlabelled = [255, 0, 0]

n_classes = 5  # было 7
images_path = 'data/res/'
input_width = 416
input_height = 608
output_path = 'imgs_results/res7/'
DataPath = 'data/'

model = 'VGGsegNet_n' + str(n_classes) + 'classes.h5'

if n_classes == 7:
    colors = np.array([Building, Grass, Development, Concrete, Roads, NotAirplanes, Unlabelled])
elif n_classes == 5:
    colors = np.array([Development, Grass, Concrete, Unlabelled, Building])


def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


def visualize(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes - 1):
        r[temp == l] = colors[l, 0]
        g[temp == l] = colors[l, 1]
        b[temp == l] = colors[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)
    rgb[:, :, 1] = (g / 255.0)
    rgb[:, :, 2] = (b / 255.0)
    return rgb


def getImageArr(path, width, height, imgNorm="sub_mean", odering='channels_first'):
    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img / 255.0

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img
    except (Exception):
        print(path)
        img = np.zeros((height, width, 3))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img


def create_predict(images_path, output_path, input_height, input_width, save_weights_path, n_classes):
    set_keras_backend("theano")

    # with open('VGGsegNet.json') as model_file:
    #     m = models.model_from_json(model_file.read())

    print(model)
    m = models.load_model(model)
    output_width = 304
    output_height = 208
    m.load_weights(save_weights_path)
    m.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    images = glob.glob(images_path + "*.png")
    images.sort()

    i = 0
    for imgName in images:

        # img = cv2.imread(imgName)
        # print(imgName)
        # hist_img = histogram_equalize(img)
        # cv2.imwrite(imgName, hist_img)

        outName = imgName.replace(images_path, output_path)
        X = getImageArr(imgName, input_height, input_width)
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        cv2.imwrite(outName, seg_img)
        i += 1